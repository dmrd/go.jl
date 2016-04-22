using go

if length(ARGS) < 2
    println(STDERR, "usage: extract.jl {cgs|kgs} output_file.hf5 {optional filename regex}")
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
    matcher = Regex(".sgf")
    filtered = Vector{AbstractString}()
    for filename in files
        if match(matcher, filename) != nothing
            push!(filtered, filename)
        end
    end
    files = filtered
    println(STDERR, "Contains filtered down to $(length(files))")
end

if DATASOURCE == "cgs"
    println(STDERR, "Finding SGF games with highly ranked players...")
    files =  filter(x -> go.players_over_rating(x, 2500), files)
end


println("Found $(length(files)) matching sgf files")

files = files[randperm(length(files))]

println(STDERR, "Extracting features from $(length(files)) games...")
go.extract_to_hdf5(OUTPUT_NAME, files, feats)
