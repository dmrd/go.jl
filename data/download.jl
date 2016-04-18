# Script to download the go archives
# Can download all with:
# download_all(cgs_archives, "outdir_path")

using Requests

# Links for current cgs game server.  More online here:
#cgs_old_url = "http://cgos.boardspace.net/9x9/archive.html"
#cgs_new_url = "http://www.yss-aya.com/cgos/9x9/archive.html"
cgs_archives = [
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2016_03.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2016_02.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2016_01.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2015_12.tar.bz2"
               "http://www.yss-aya.com/cgos/9x9/archives/9x9_2015_11.tar.bz2"
               ]

"Fetch the urls for all the kgs game records"
function fetch_kgs_urls()
    # Download page source & parse out the record names
    kgs_url = "http://www.u-go.net/gamerecords/"
    response = get(kgs_url)
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

"Download file and optionally untar/bz2 and remove archive"
function download_archive(url, outdir; expand=false, remove=false)
    !ispath(outdir) && mkdir(outdir)
    name = ascii(split(url, "/")[end])
    outpath = joinpath(outdir, name)

    archive = get(url)
    save(archive, outpath)  # Save the payload to a file
    if expand
        @osx? run(`tar xfz $(outpath) -C $(outdir)`) : run(`tar xjf $(outpath) -C $(outdir)`)
    end
    if remove
        run(`rm $(outpath)`)
    end
end

function download_all(urls, outdir)
    ts = time()
    for (i,url) in enumerate(urls)
        println("$(i)/$(length(urls)): $(url)")
        download_archive(url, outdir, expand=true, remove=true)
    end
    println("Took $(time() - ts) seconds")
end
