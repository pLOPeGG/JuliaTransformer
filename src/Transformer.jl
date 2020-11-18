### A Pluto.jl notebook ###
# v0.11.14

module Transformer

using Random

using Flux
using Flux: softmax, glorot_uniform, Dense, gelu, LayerNorm, trainable, Dropout

using Plots


export TransformerModel, TransformerEncoder

struct Attention
	d_model
	d_proj
	W_q
	W_k
	W_v
end

Attention(d_model::Integer, d_proj::Integer) = 
	Attention(d_model,
				d_proj,
				glorot_uniform(d_proj, d_model),
				glorot_uniform(d_proj, d_model), 
				glorot_uniform(d_proj, d_model))



struct MultiheadAttention
	d_model
	d_proj
	n_head
	attn_heads
end

function MultiheadAttention(d_model::Integer, n_head::Integer)
	@assert d_model % n_head == 0
	MultiheadAttention(d_model,
						d_model ÷ n_head,
						n_head,
						[Attention(d_model, d_model ÷ n_head) for _ in 1:n_head])
end


struct FeedForward
	d_model
	d_proj
	dense_α
	dense_β
end

FeedForward(d_model, d_proj) = FeedForward(d_model,
											d_proj,
											Dense(d_model, d_proj, gelu),
											Dense(d_proj, d_model))



struct PositionalEncoding
	d_model
	dropout
	max_len
	pe
end

function compute_positional_encoding(d_model, max_len)
	pe = zeros(d_model, max_len)
	positions = 0.:max_len-1
	div_term = exp.((0.:2:d_model-1) .* (-log(10_000) / d_model))

	pe[1:2:end, :] = sin.(div_term * positions')
	pe[2:2:end, :] = cos.(div_term * positions')
	pe
end

PositionalEncoding(d_model, drop=0.1; max_len=512) =
	PositionalEncoding(d_model,
						Dropout(drop),
						max_len,
						compute_positional_encoding(d_model, max_len))

struct Embedding
	d_in
	d_out
	w
end

Embedding(d_in, d_out) = Embedding(d_in, d_out, glorot_uniform(d_out, d_in))


struct TransformerEncoderLayer
	d_model
	multihead_attn
	norm_attn
	feed_forward
	norm_ffwd
end

TransformerEncoderLayer(d_model::Integer, n_head::Integer, d_proj_ffwd::Integer) =
	TransformerEncoderLayer(d_model,
							MultiheadAttention(d_model, n_head),
							LayerNorm(d_model),
							FeedForward(d_model, d_proj_ffwd),
							LayerNorm(d_model))



struct TransformerEncoder
	d_model
	n_layer
	layers
end

TransformerEncoder(d_model::Integer, n_layer::Integer, n_head::Integer, d_proj_ffwd::Integer) =
	TransformerEncoder(d_model,
						n_layer,
						[TransformerEncoderLayer(d_model, n_head, d_proj_ffwd)
						for _ in 1:n_layer])


struct TransformerModel
	encoder::TransformerEncoder
	embeddings::Embedding
	pos_encoding::PositionalEncoding
end

TransformerModel(encoder::TransformerEncoder, vocab_size::Integer) = TransformerModel(encoder::TransformerEncoder, Embedding(vocab_size, encoder.d_model), PositionalEncoding(encoder.d_model, ))

# Defining trainable parameters

Flux.trainable(m::Attention) = (m.W_q, m.W_k, m.W_v,)

Flux.trainable(m::MultiheadAttention) = (m.attn_heads,)

Flux.trainable(m::FeedForward) = (m.dense_α, m.dense_β,)

Flux.trainable(m::PositionalEncoding) = ()

Flux.trainable(m::Embedding) = (m.w,)

Flux.trainable(m::TransformerEncoderLayer) = (m.multihead_attn, m.norm_attn, m.feed_forward, m.norm_ffwd,)

Flux.trainable(m::TransformerEncoder) = (m.layers,)

Flux.trainable(m::TransformerModel) = (m.encoder, m.embeddings)




# Implementations of layers
(m::Attention)(x) = m(x, x, x)

(m::Attention)(q, k, v) = softmax((m.W_q * q * (m.W_k * k)') ./ sqrt(m.d_proj)) * (m.W_v * v)

(m::MultiheadAttention)(x) = vcat([attn(x) for attn in m.attn_heads]...)

(m::FeedForward)(x) = x |> m.dense_α |> m.dense_β

(m::PositionalEncoding)(x) = x + m.pe[:, 1:size(x, 2)] |> m.dropout

(m::Embedding)(x) = m.w * x

function (m::TransformerEncoderLayer)(x)
	y = (x |> m.multihead_attn) + x |> m.norm_attn
	z = (y |> m.feed_forward) + y |> m.norm_ffwd
	z
end

function (m::TransformerEncoder)(x)
	for layer in m.layers
		x = x |> layer
	end
	x
end

function (m::TransformerModel)(x)
	y = x |> m.embeddings |> m.pos_encoding
	z = inverse_embedding(m.embeddings, y |> m.encoder) |> softmax
	z
end

function inverse_embedding(e::Embedding, x)
	e.w' * x
end

heatmap(PositionalEncoding(512).pe')

transformer = TransformerEncoder(10, 3, 2, 15)


x = rand(10, 20)
x |> transformer

end  # end module Transformer