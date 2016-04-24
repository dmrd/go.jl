# Similar to main trainer script, but focused on using a generator to provide data to Keras
# SGF generator is slightly slower than HF5 because of overhead

using go
using ArgParse
using HDF5


# Some flags are used for different purposes for h5 and loading sgf
function parseargs()
    s = ArgParseSettings()
    @add_arg_table s begin
        "model_name"
          help = "Model name (saved in models dir)"
          arg_type = AbstractString
          required = true
        "--h5path"
          help = "Where to load h5data from"
          default = nothing
        "--kgs"
          help = "Load data from kgs sgf files"
          action = :store_true
        "--cgs"
          help = "Load data from cgs sgf files"
          action = :store_true
        "--epochs"
          help = "Number training epochs"
          arg_type = Int
          default = 10
        "--n_train"
          help = "Maximum number of sgf files to load OR #moves from h5"
          arg_type = Int
          default = 10000
        "--batch_size"
          help = "Batch size"
          arg_type = Int
          default = 128
        "--n_per_epoch"
          help = "Number of examples per epoch"
          arg_type = Int
          default = 12800
        "--n_validation"
          help = "How many files to use for validation OR # moves from h5"
          arg_type = Int
          default = 100
        "--load"
          help = "Load weights when trained model already exists"
          action = :store_true
    end
    parse_args(s)
end

pargs = parseargs()

# Set the feature extractors
FEATS = go.LIBERTIES

LOAD_H5 = pargs["h5path"] != nothing

nval = pargs["n_validation"]
ntrain = pargs["n_train"]
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

    if cgs
        println(STDERR, "Finding SGF games with highly ranked players...")
        files =  filter(x -> go.players_over_rating(x, pargs["rating"]), files)
    end

    println("Found $(length(files)) matching sgf files")
    randomized = files[randperm(length(files))]

    validation_files = randomized[1:nval]
    train_files = randomized[nval + 1:min(end, nval + ntrain)]

    if length(train_files) < ntrain
        println(STDERR, "Not enough files for both train and validation")
    end

    train_generator = go.SGFGenerator(train_files, FEATS)

    println(STDERR, "Extracting validation examples...")
    val_data = go.generate_training_data(validation_files; progress_update=500, features=FEATS)
    valX = go.batch_training_examples([x[1] for x in val_data])
    valY = go.batch_training_examples([x[2][:] for x in val_data])
    @show(size(valX))
    @show(size(valY))
    val_data=nothing
    println(STDERR, "Validating on $(size(valX)[end]) moves")
else
    f = h5open(pargs["h5path"])
    hfN = size(f["data"])[end]

    # Load validation set into memory so there's less stuff to deal with...
    # val_generator = go.HF5Generator(f, FEATS, pargs["batch_size"],
    #                                 1, nval)
    valX = f["data"][:,:,:,1:nval]
    valY = f["label"][:,1:nval]
    close(f)
    train_generator = go.HF5Generator(pargs["h5path"], FEATS, pargs["batch_size"],
                                      nval + 1,
                                      min(nval + 1 + ntrain, hfN))
end

println(STDERR, "Initializing model...")
clf = go.CLF_LINEAR(FEATS)

println(STDERR, "Training...")
tm = time()
MODEL_NAME = pargs["model_name"]
checkpoint_path = "../models/$(MODEL_NAME)_checkpoint.hf5"
go.save_model(clf, "../models/", MODEL_NAME, save_weights=false, save_yaml=true)

if pargs["load"]
    clf.model.load_weights("../models/$(MODEL_NAME).hf5")
end

go.keras_train_generator(clf, train_generator,
                         epochs=pargs["epochs"],
                         samples_per_epoch=pargs["n_per_epoch"],
                         validation_data=(go.to_python_array(valX), go.to_python_array(valY)),
                         checkpoint_path=checkpoint_path)

println("Took $(time() - tm) seconds")

println(STDERR, "Saving model...")
go.save_model(clf, "../models/", MODEL_NAME, save_weights=true, save_yaml=false, overwrite=pargs["load"]) 
