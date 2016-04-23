using go

if length(ARGS) < 2
    println(STDERR, "usage: extract.jl {cgs|kgs} output_file.hf5 {optional filename regex} {max_files}")
    exit()
end

DATASOURCE = ARGS[1]
OUTPUT_NAME = ARGS[2]

# Set the feature extractors
feats = go.LIBERTIES

println(STDERR, "Finding SGF files...")
files = go.find_sgf("../data/$(DATASOURCE)/");
println(STDERR, "Found $(length(files)) sgf files")

# Filter to filenames containing regex (e.g. month/year)
if length(ARGS) == 3
    matcher = Regex(ARGS[3])
    filtered = Vector{AbstractString}()
    for filename in files
        if match(matcher, filename) != nothing
            push!(filtered, filename)
        end
    end
    files = filtered
    println(STDERR, "Contains filtered down to $(length(files))")
end

println(STDERR, "SGF games with highly ranked players...")
if DATASOURCE == "cgs"
    files =  filter(x -> go.players_over_rating(x, 2000), files)
#elseif contains(DATASOURCE, "kgs")
#    files =  filter(x -> go.players_over_rating(x, 7, dan=true), files)
end

if length(ARGS) == 4 
    maxfiles = parse(Int, ARGS[4])
    if maxfiles < length(files)
        files = files[randperm(length(files))][1:maxfiles]
    end
end

println("Found $(length(files)) matching sgf files")

files = files[randperm(length(files))]

println(STDERR, "Extracting features from $(length(files)) games...")
go.extract_to_hdf5(OUTPUT_NAME, files, feats, nlabels = go.N*go.N)
