haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Documenter
using KadanoffBaym

makedocs(
    sitename = "KadanoffBaym: The User Guide",
    clean = true,
    authors = "Li Huang <huangli@caep.cn> and contributors",
    format = Documenter.HTML(
        prettyurls = false,
        ansicolor = true,
        repolink = "https://github.com/huangli712/KadanoffBaym",
        size_threshold = 409600, # 400kb
        assets = ["assets/kadanoffbaym.css"],
        collapselevel = 1,
    ),
    #format = Documenter.LaTeX(platform = "none"),
    remotes = nothing,
    modules = [KadanoffBaym],
    pages = [
        "Welcome" => "index.md",
        "Library" => Any[
            "KadanoffBaym" => "library/kadanoffbaym.md",
            "Constants" => "library/global.md",
            "Types" => "library/type.md",
            "Integration Weights" => "library/weight.md",
            "Utilities" => "library/util.md",
        ],
    ],
)
