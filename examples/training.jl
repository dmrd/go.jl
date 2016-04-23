using go
using ArgParse


# Some flags are used for different purposes for h5 and loading sgf
function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "model_name"
          help = "Model name (saved in models dir)"
          arg_type = AbstractString
          required = true
        "--h5path"
          help = "Path to h5 file to load training data from"
          arg_type = AbstractString
        "--kgs"
          help = "Load data from kgs sgf files"
          action = :store_true
        "--cgs"
          help = "Load data from cgs sgf files"
          action = :store_true
        "--rating"
          help = "Minimum elo rating (for cgs)"
          default = 2500
          arg_type = Int
        "--epochs"
          help = "Number training epochs"
          arg_type = Int
          default = 10
        "--filename_substring"
          help = "Filter down to sgf files which contain this substring"
          arg_type = AbstractString
          default = ".sgf"
        "--n_train"
          help = "Maximum number of sgf files to load OR how many moves to load from h5 file"
          arg_type = Int
          default = 10000000  # Basically no limit
        "--n_validation"
          help = "How many moves to use for validation in h5"
          arg_type = Int
          default = 10000
    end
    parse_args(s)
end

pargs = parseargs()

# Set the feature extractors
feats = go.LIBERTIES

LOAD_H5 = pargs["h5path"] != nothing

if !LOAD_H5
    cgs = pargs["cgs"]
    kgs = pargs["kgs"]
    if (kgs + cgs) != 1
        println(STDERR, "Exactly one of h5path, cgs, or kgs must be set")
        exit()
    end
    DATASOURCE = cgs ? "cgs" : "kgs"
    println(STDERR, "Finding SGF files...")
    files = go.find_sgf("../data/$(DATASOURCE)/");
    println(STDERR, "Found $(length(files)) sgf files")

    # Filter to filenames containing regex (e.g. month/year)
    if "filename_substring" in keys(pargs)
        matcher = Regex(pargs["filename_substring"])
        filtered = Vector{AbstractString}()
        for filename in files
            if match(matcher, filename) != nothing
                push!(filtered, filename)
            end
        end
        files = filtered
        println(STDERR, "Contains filtered down to $(length(files))")
    end

    if cgs
        println(STDERR, "Finding SGF games with highly ranked players...")
        files =  filter(x -> go.players_over_rating(x, pargs["rating"]), files)
    end
    # if kgs...

    println("Found $(length(files)) matching sgf files")

    files = files[randperm(length(files))][1:min(end, pargs["n_train"])]

    println(STDERR, "Extracting features from $(length(files)) games...")
    examples = go.generate_training_data(files; progress_update=500, features=feats)
    X = go.batch_training_examples([x[1] for x in examples])
    Y = go.batch_training_examples([x[2][:] for x in examples])
end

println(STDERR, "Initializing model...")
clf = go.CLF_LINEAR(feats)

println(STDERR, "Training...")
tm = time()
MODEL_NAME = pargs["model_name"]
checkpoint_path = "../models/$(MODEL_NAME)_checkpoint.hf5"
go.save_model(clf, "../models/", MODEL_NAME, save_weights=false, save_yaml=true)

if LOAD_H5
    go.keras_train_h5(clf, pargs["h5path"],
                      epochs=pargs["epochs"],
                      n_train=pargs["n_train"],
                      n_validation=pargs["n_validation"],
                      checkpoint_path=checkpoint_path)
else
    go.train_model(clf, X, Y, epochs=pargs["epochs"], validation_split=0.05, checkpoint_path=checkpoint_path)
end
println("Took $(time() - tm) seconds")

println(STDERR, "Saving model...")
go.save_model(clf, "../models/", MODEL_NAME, save_weights=true, save_yaml=false) 
