module PreTraining

using Random
using Flux

function apply_mask(text_tokens::Array{String}, mask_token::String; proportion=0.15)
    masked_tokens = text_tokens |> copy
    for (i, tok) in text_tokens |> enumerate
        rnd = Random.rand()
        if rnd < proportion
            masked_tokens[i] = mask_token
        end
    end
    masked_tokens, text_tokens
end


function apply_mask(text_tokens::Array{Integer}, mask_token::Integer; proportion=0.15)
    masked_tokens = text_tokens |> copy
    for (i, tok) in text_tokens |> enumerate
        rnd = Random.rand()
        if rnd < proportion
            masked_tokens[i] = mask_token
        end
    end
    masked_tokens, text_tokens
end


struct DataLoader{D}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}



end