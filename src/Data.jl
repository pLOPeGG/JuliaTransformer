module Data

using Unicode

export get_texts, normalize_text

function get_texts()
    all_texts = []
    files = readdir(raw"C:\Users\Doudou\Documents\Code\JuliaTransformer\ELTeC-fra\plain1")
    for file in files
        open(raw"C:\Users\Doudou\Documents\Code\JuliaTransformer\ELTeC-fra\plain1\\" * file) do io
            text = readlines(io)
            push!(all_texts, join(text))
        end
    end
    all_texts
end

function normalize_text(text)
	text = Unicode.normalize(text, stripmark=true, decompose=true)
	d_replace = Dict(
		r"æ" => s"ae",
		r"Æ" => s"Ae",
		r"œ" => s"oe",
		r"Œ" => s"Oe",
		r"(–|—|—|\u00ad)" => s"-",
		r"…" => s"...",
		r"(·|•|■)" => s".",
		r"(«|»|“|”|˝|˵)" => s"\"",
		r"(`̀|´|‘|’)" => s"'",
		r"‹" => s"<",
		r"›" => s">",
		r"(°|∘|º|ø)" => s"o",
		r"(?<=\w)([.,:;?!])" => s" \1",
		r"(⁂|↑)" => s""
	)
	for (r, s) in d_replace
		text = replace(text, r=>s)
	end
	text
end

end