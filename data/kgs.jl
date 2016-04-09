# Script to download the kgs archives
# Use from repl

using Requests

function fetch_names()
    url = "http://www.u-go.net/gamerecords/"
    # Download page source & parse out the record names
    response = get(url)
    html = readall(response)
    fileurls = Vector{AbstractString}()
    regex = r"(dl\..*\.tar\.bz2)"
    offset = 1
    while true
        result = match(regex, html, offset)
        if result != nothing
            push!(fileurls, string("http://", result.match))
            offset = result.offset + 1
        else
            break
        end
    end
    fileurls
end


function download_archive(url, outdir;expand=false, remove=false)
    name = ascii(split(url, "/")[end])
    outpath = joinpath(outdir, name)

    archive = get(url)
    save(archive, outpath)  # Save the payload to a file
    if expand
        run(`tar xfz $(outpath)`)
    end
    if remove
        run(`rm $(outpath)`)
    end
end

function run_all(outdir)
    names = fetch_names()
    n = length(names)
    ts = time()
    for (i,url) in enumerate(names)
        println("$(i)/$(n): $(url)")
        download_archive(url, outdir, expand=true, remove=true)
    end
    println("Took $(time() - ts) seconds")
end
