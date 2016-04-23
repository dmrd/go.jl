using go

if length(ARGS) < 4
    println(STDERR, "usage: training.jl {cgs|kgs} MODEL_NAME num_epochs nexamples (optional filename substring)")
    exit()
end

DATASOURCE = ARGS[1]
MODEL_NAME = ARGS[2]
EPOCHS = parse(Int, ARGS[3])
NUM_EXAMPLES = parse(Int, ARGS[4])

# Set the feature extractors
feats = go.LIBERTIES

println(STDERR, "Finding SGF files...")
files = go.find_sgf("../data/$(DATASOURCE)/");
println(STDERR, "Found $(length(files)) sgf files")

# Filter to filenames containing regex (e.g. month/year)
if length(ARGS) == 5
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

files = rand(files, NUM_EXAMPLES)

println(STDERR, "Extracting features from $(length(files)) games...")
examples = go.generate_training_data(files; progress_update=500, features=feats)
X = go.batch_training_examples([x[1] for x in examples])
Y = go.batch_training_examples([x[2][:] for x in examples])

println(STDERR, "Initializing model...")
clf = go.CLF_DCNN(feats)

println(STDERR, "Training...")
tm = time()
checkpoint_path = "../models/$(MODEL_NAME)_checkpoint.hf5"
go.save_model(clf, "../models/", MODEL_NAME, save_weights=false, save_yaml=true) 
go.train_model(clf, X, Y, epochs=EPOCHS, validation_split=0.05, checkpoint_path=checkpoint_path)
println("Took $(time() - tm) seconds")

println(STDERR, "Saving model...")
go.save_model(clf, "../models/", MODEL_NAME, save_weights=true, save_yaml=false) 
