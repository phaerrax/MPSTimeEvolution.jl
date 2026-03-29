using Documenter, DocumenterCitations
using MPSTimeEvolution

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    modules=[MPSTimeEvolution],
    sitename="MPSTimeEvolution",
    repo=Remotes.GitHub("phaerrax", "MPSTimeEvolution.jl"),
    checkdocs=:exported,
    authors="Davide Ferracin and Dario Tamascelli",
    pages=[
        "Home" => "index.md",
        "Reference" => "reference.md",
        "Callback objects" => "callback_obj.md",
        "Tutorial" => ["examples/tdvp1.md", "examples/tdvp1vec.md"],
    ],
    plugins=[bib],
    format=Documenter.HTML(;
        mathengine=Documenter.MathJax3(
            Dict(
                :tex => Dict(
                    :inlineMath => [["\$", "\$"], [raw"\(", raw"\)"]],
                    :tags => "ams",
                    :packages =>
                        ["base", "ams", "autoload", "configmacros", "mathtools"],
                    :macros => Dict(
                        :C => [raw"\mathbb{C}"],
                        :Im => [raw"\mathrm{Im}"],
                        :N => [raw"\mathbb{N}"],
                        :Num => [raw"\mathscr{N}"],
                        :R => [raw"\mathbb{R}"],
                        :Re => [raw"\mathrm{Re}"],
                        :abs => [raw"\lvert #1 \rvert", 1],
                        :adj => [raw"#1^\dagger", 1],
                        :avg => [raw"\langle #1\rangle", 1],
                        :blank => [raw"{-}"],
                        :bra => [raw"\langle #1 \rvert", 1],
                        :conj => [raw"\bar{#1}", 1],
                        :dd => [raw"\mathrm{d}"],
                        :defeq => [raw"\mathrel{\mathop:}="],
                        :det => [raw"\operatorname{det}"],
                        :diag => [raw"\operatorname{diag}"],
                        :dissipator => [raw"\mathscr{D}"],
                        :fcomm => [raw"\{#1\}", 1],
                        :id => [raw"1"],
                        :imag => [raw"_{\mathrm{i}}"],
                        :imat => [raw"I_{#1}", 1, ""],
                        :innp => [raw"\langle #1, #2\rangle", 2],
                        :iu => [raw"\mathrm{i}"],
                        :ket => [raw"\lvert #1 \rangle", 1],
                        :lindblad => [raw"\mathscr{L}"],
                        :norm => [raw"\lVert #1 \rVert", 1],
                        :outp => [raw"\lvert #1\rangle \langle #2\rvert", 2],
                        :paulim => [raw"\sigma_-^{(#1)}", 1, ""],
                        :paulip => [raw"\sigma_+^{(#1)}", 1, ""],
                        :paulix => [raw"\sigma_x^{(#1)}", 1, ""],
                        :pauliy => [raw"\sigma_y^{(#1)}", 1, ""],
                        :pauliz => [raw"\sigma_z^{(#1)}", 1, ""],
                        :real => [raw"_{\mathrm{r}}"],
                        :sb => [raw"_{#1}", 1],
                        :set => [raw"\{\, #1 \;\vert\; #2\,\}", 2],
                        :spinup => [raw"\uparrow"],
                        :spindown => [raw"\downarrow"],
                        :tr => [raw"\operatorname{tr}"],
                        :transpose => [raw"#1^{\mathrm{T}}", 1],
                        :tsp => [raw"^{\otimes #1}", 1],
                        :vacuum => [raw"\varOmega"],
                    ),
                ),
            ),
        ),
    ),
)

# Automatically deploy documentation to gh-pages.
deploydocs(; repo="github.com/phaerrax/MPSTimeEvolution.jl.git")
