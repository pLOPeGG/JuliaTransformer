module PreTraining

using Random

function apply_mask(text_tokens::Array{Int}, mask_token::Int; proportion::Real = 0.15)
    masked_tokens = text_tokens |> copy
    for (i, tok) in text_tokens |> enumerate
        rnd = Random.rand()
        if rnd < proportion
            masked_tokens[i] = mask_token
        end
    end
    masked_tokens, text_tokens
end

struct LiveDataLoader{D}
    data::D  # whole text
    batch_size::Int
    sentence_size::Int
    nobs::Int
    shuffle::Bool
    mask_token::Int
    mask_prop::Real
end

function Base.iterate(iter::LiveDataLoader, state=nothing)
    if isnothing(state)
        state = 1
    end
    (state > iter.nobs) && return nothing

    beg_indices = rand(1:(length(iter.data) - iter.sentence_size), iter.batch_size)
    batch_tgt = [iter.data[r:r+iter.sentence_size-1] |> collect for r in beg_indices]
    batch_in, batch_out = zip(apply_mask.(batch_tgt, iter.mask_token, proportion=iter.mask_prop)...)

    return ((batch_in, batch_out), state + 1)
end



end