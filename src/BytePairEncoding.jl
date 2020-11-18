### A Pluto.jl notebook ###
# v0.12.4
module BytePairEncoding

using Markdown

md"""
## TODO:

- [x]  Speedup by not join / split all the time
- [x]  Implement String Tree for vocab
- [x]  Implement Tokenizer (use StringTree)
- [x]  Progress Bar in BPE train iter / size
- [x]  Split punctuation before BPE
- [x]  Vocabulary type (
		+ add index to each stop in Trie
		+ fast reverse dict
		+ tokenize to index
						)
"""

export Vocabulary, tokenize, tokenize_index, normalize_text

using BenchmarkTools
using DataStructures: Deque, OrderedDict
using Unicode
using ProgressMeter

include("Data.jl")
using .Data


md"""
# Byte Pair Encoding algorithm
"""

all_texts = get_texts()


function get_tokens(text)
	words = text |> split
	
	word_counter = Dict()
	for w in words
		word_tokens = [["__"] ; string.(w|>collect)]
		word_counter[word_tokens] = get!(word_counter, word_tokens, 0) + 1
	end
	word_counter
end

function get_pairs(tokens_count)
	pair_counter = Dict()
	for (word, freq) in tokens_count
		symbols = word
		
		for i in 1:length(symbols)-1
			cur, nxt = symbols[i], symbols[i+1]
			pair_counter[(cur, nxt)] = get!(pair_counter, (cur, nxt), 0) + freq
		end
	end
	pair_counter
end

function merge(pair::Tuple{String, String}, tokens_count)
	new_tokens_count = Dict()
	for (word, freq) in tokens_count
		tokens = word
		new_word = []
		skip_next = false
		for (i, token) in enumerate(tokens[1:end-1])
			if skip_next
				skip_next = false
				continue
			end
			if (token, tokens[i+1]) == pair
				push!(new_word, token * tokens[i+1])
				skip_next = true
			else
				push!(new_word, token)
			end
		end
		if !skip_next
			push!(new_word, tokens[end])
		end
		new_tokens_count[new_word] = freq
	end
	new_tokens_count
end

function merge(pairs::Array{Tuple{String, String}}, tokens_count)
	new_tokens_count = Dict()

	pairs_hash = hash.(pairs) |> Set

	for (word, freq) in tokens_count
		tokens = word
		new_word = []
		skip_next = false
		for (i, token) in enumerate(tokens[1:end-1])
			if skip_next
				skip_next = false
				continue
			end
			if (token, tokens[i+1]) |> hash in pairs_hash
				push!(new_word, token * tokens[i+1])
				skip_next = true
			else
				push!(new_word, token)
			end
		end
		if !skip_next
			push!(new_word, tokens[end])
		end
		new_tokens_count[new_word] = freq
	end
	new_tokens_count
end

function unique_tokens(tokens_count, base_vocab=nothing)
	
	if isnothing(base_vocab)
		base_vocab = Set{String}()
	end
	for (splitted_word, freq) in tokens_count
		union!(base_vocab, splitted_word)
	end
	base_vocab
end

function bpe_step(tokens; n::Int=1)
	pairs = get_pairs(tokens)
	most_freq_pairs = first.(partialsort(pairs |> collect, 1:n, by=(p) -> p.second, rev=true))
	merge(most_freq_pairs, tokens)
end

function bpe_iter(text, iter=50; batch_merge=1)
	tokens = get_tokens(text)
	@showprogress for i in 1:iter
		if batch_merge == 1
			tokens = tokens |> bpe_step
		else
			tokens = bpe_step(tokens, n=batch_merge)
		end
	end
	tokens
end

function bpe_vocab_size(text, vocab_size=1000; max_iter=1000, batch_merge=1)
	tokens = get_tokens(text)
	
	get_token_size(t) = t |> unique_tokens |> length
	token_size = tokens |> get_token_size
	new_token_size = token_size

	p = Progress(vocab_size - token_size, 1)
	i = 1
	while i <= max_iter && token_size < vocab_size
		if batch_merge == 1
			tokens = tokens |> bpe_step
		else
			tokens = bpe_step(tokens, n=batch_merge)
		end
		
		new_token_size = tokens |> get_token_size
		next!(p, step=new_token_size - token_size)
		token_size = new_token_size

		i += 1
	end
	tokens
end

struct Trie
	cum_string
	children
	n_reserved_tokens
end

mutable struct Node
	char
	cum_string
	stop
	token_index
	children
end

Trie(;n_reserved::Integer = 100) = Trie("", OrderedDict(), n_reserved)
Trie(tokens::AbstractArray{String} ;n_reserved::Integer = 100) = build_trie(tokens, n_reserved=n_reserved)
Trie(tokens::AbstractSet{String}; n_reserved::Integer = 100) = build_trie(tokens, n_reserved=n_reserved)
Trie(file::String; n_reserved::Integer = 100) = open(file, "r") do io
		Trie(readlines(io) |> collect, n_reserved=n_reserved)
	end
Base.get(t::Trie, k, default) = get(t.children, k, default)
function Base.get!(t::Trie, key::AbstractString, default::Node)
	curr_node = t
	for c in key[1:end-1]
		curr_node = get!(curr_node.children, c, Node(c, curr_node.cum_string * c, false))
	end
	get!(curr_node.children, key[end], default)
end

Base.getindex(t::Trie, i...) = getindex(t.children, i...)
function Base.getindex(t::Trie, s::String)
	n = t
	for c in s
		n = getindex(n.children, c)
	end
	n
end
Base.haskey(t::Trie, k) = haskey(t.children, k)
Base.length(t::Trie) = vocab_size(t)

function Base.collect(t::Trie)
	tokens = String[]
	for (k, n) in t.children
		append!(tokens, n |> collect)
	end
	tokens
end

Node(c::Char, s::String, stop::Bool, token_index::Integer = -1) = Node(c, s, stop, token_index, OrderedDict())
Base.get(n::Node, k, default) = get(n.children, k, default)
Base.getindex(n::Node, i...) = getindex(n.children, i...)
Base.haskey(n::Node, k) = haskey(n.children, k)

function Base.collect(n::Node)
	tokens = String[]
	if n.stop
		push!(tokens, n.cum_string)
	end
	for (k, m) in n.children
		append!(tokens, m |> collect)
	end
	tokens
end

struct Vocabulary
	trie::Trie
	tokens::Array{String}  # [special tokens || trie |> collect]

	function Vocabulary(t::Trie, special_tokens::Array{String})
		if t.n_reserved_tokens >= length(special_tokens)
			tokens = vcat(special_tokens, ["__rsv$(i)__" for i in 1:(t.n_reserved_tokens-length(special_tokens))], t |> collect)
			
			for (i, tok) in enumerate(special_tokens)
				node = get!(t, tok, Node(tok[end], tok, true, i))
				@assert (node.token_index == i) "Special token $tok already in Trie"
			end
		else
			@assert (vocab_size(t) == length(special_tokens)) "Trie size and tokens length missmatch : $(vocab_size(t)) != $(length(special_tokens))"
			tokens = special_tokens
		end
		new(t, tokens)
	end
end

function Vocabulary(text::AbstractString, vocab_size::Integer, tokens::AbstractArray{String}; max_iter=1000, batch_merge=1)
	trie = run_bpe(text, vocab_size, max_iter=max_iter, batch_merge=batch_merge)
	Vocabulary(trie, tokens)
end

Base.getindex(vocab::Vocabulary, i::Integer...) = getindex(vocab.tokens, i...)
Base.getindex(vocab::Vocabulary, i::AbstractString...) = getindex(vocab.trie, i...)
Base.getindex(vocab::Vocabulary, i::AbstractChar...) = getindex(vocab.trie, i...)


function Vocabulary(file::AbstractString)
	open(file, "r") do io
		n_reserved = parse(Int64, readline(io))
		tokens = readlines(io)
		trie = Trie(tokens[n_reserved+1:end], n_reserved=n_reserved)

		for (i, tok) in enumerate(tokens[1:n_reserved])
			node = get!(trie, tok, Node(tok[end], tok, true, i))
			@assert (node.token_index == i) "Special token $tok already in Trie"
		end
		Vocabulary(trie, tokens)
	end
end

function save(vocab::Vocabulary, file::String)
	open(file, "w") do io
		write(io, "$(vocab.trie.n_reserved_tokens)\n")
		write(io, join(vocab.tokens, "\n"))
	end
end

function build_trie(unique_tokens; n_reserved::Integer = 100)
	trie = Trie(n_reserved=n_reserved)

	for token in sort(unique_tokens |> collect)
		node = trie
		for c in token
			node = get!(node.children, c, Node(c, node.cum_string * c, false))
		end
		node.stop = true
	end

	ordered_tokens = trie |> collect
	for (i, token) in enumerate(ordered_tokens)
		trie[token].token_index = i + n_reserved
	end
	
	trie
end

function vocab_size(t::Trie)
	s = 0
	for (k, n) in t.children
		s += vocab_size(n)
	end
	s
end

function vocab_size(n::Node)
	s = n.stop ? 1 : 0
	for (k, m) in n.children
		s += vocab_size(m)
	end
	s
end

function parse_word(trie, word; new_word=true)
	tokens = []
	i = 1
	last_pos = 0
	
	if new_word
		curr_node = trie["__"]
		last_token = curr_node.cum_string
	else
		curr_node = trie
		last_token = nothing
	end
	while i <= length(word)
		char = word[i]
		if haskey(curr_node, char)
			curr_node = curr_node[char]
		else
			push!(tokens, last_token)
			i = last_pos + 1
			char = word[i]
			
			if haskey(trie, char)
				curr_node = trie[char]
			else
				@warn "Char '$char' is not included in trie"
				# TODO: Push UNK token instead of skipping
				# push!(tokens, )
				curr_node = trie
			end
			last_token = nothing
		end
		if curr_node.stop
			last_token = curr_node.cum_string
			last_pos = i
		end
		i += 1
	end

	if curr_node.stop
		push!(tokens, curr_node.cum_string)
	elseif !isnothing(last_token)
		push!(tokens, last_token)
		append!(tokens, parse_word(trie, word[last_pos+1:end], new_word=false))
	else
		throw(ValueError())
	end
	tokens
end

function parse_word_opti(trie, word)
	# Build acyclic graph
	graph = Dict()
	n = word |> length
	for i in 0:n
		j = i + 1
		node = trie
		if i == 0
			node = node["__"]
		end
		while j <= n && haskey(node, word[j])
			node = node[word[j]]
			if node.stop
				push!(get!(graph, i, []), (j, node.cum_string))
			end
			
			j += 1
		end
	end
	
	# BFS
	queue = Deque{Tuple{Int, Int, Int, String}}()
	push!(queue, (0, 0, -1, ""))
	
	path = Dict()
	
	while !isempty(queue)
		(i, d, prev, tok) = popfirst!(queue)
		
		if !haskey(path, i)
			path[i] = (prev, tok)
		end
		if i == n
			break
		end
		
		for (nei, tok) in get!(graph, i, [])
			if haskey(path, nei)
				continue
			end
			push!(queue, (nei, d+1, i, tok))
		end
	end
	
	tokens = []
	i = n
	while i > 0
		i, tok = path[i]
		push!(tokens, tok)
	end
	return tokens |> reverse
	
end

function tokenize(trie::Trie, text; parse_f=parse_word)
	tokens = String[]
	for word in text |> split
		append!(tokens, parse_f(trie, word))
	end
	tokens
end

tokenize(vocab::Vocabulary, text; parse_f=parse_word) = tokenize(vocab.trie, text, parse_f=parse_f)

function tokenize_index(trie::Trie, text; parse_f=parse_word)
	get_index(w) = trie[w].token_index
	tokens = Int64[]
	for word in text |> split
		append!(tokens, get_index.(parse_f(trie, word)))
	end
	tokens
end

tokenize_index(vocab::Vocabulary, text; parse_f=parse_word) = tokenize_index(vocab.trie, text, parse_f=parse_f)

rev_tokenize_index(vocab::Vocabulary, token_ids::Array{<:Integer}) = [vocab[i] for i in token_ids]

function index_to_tokens(vocab::Vocabulary, indices::Array{<:Integer})
	tokens = similar(indices, String)
	for (i, idx) in enumerate(indices)
		tokens[i] = vocab.tokens[idx]
	end
	tokens
end

function run_bpe(text, vocab_size; max_iter=1000, batch_merge=1)
	tokens_count = bpe_vocab_size(text, vocab_size; max_iter=max_iter, batch_merge=batch_merge)
	tokens = union(unique_tokens(tokens_count), string.(text |> graphemes |> collect))
	Trie(tokens)
end

vocab = Vocabulary(all_texts[1:10] |> join |> normalize_text, 1000, ["__beg__", "__end__", "__msk__", "__sep__", "__unk__"], batch_merge=20)
println(tokenize(vocab, all_texts[2] |> normalize_text)[1:100])

end  # End module BPE

