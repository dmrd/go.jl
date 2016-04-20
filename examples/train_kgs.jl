using go

if length(ARGS) != 2
    println(STDERR, "usage: training.jl MODEL_NAME num_epochs")
    exit()

end
MODEL_NAME = ARGS[1]
epochs = parse(Int, ARGS[2])

# Set the feature extractors
feats = go.LIBERTIES

println(STDERR, "Finding SGF files...")
files = go.find_sgf("../data/kgs");

# Filter to filenames containing regex (e.g. month/year)
matcher = r".sgf"
filtered = Vector{AbstractString}()
for filename in files
    if match(matcher, filename) != nothing
        push!(filtered, filename)
    end
end

println("Found $(length(filtered)) matching sgf files")
filtered = rand(filtered, 1500)
println("Training on $(length(filtered)) games")

println(STDERR, "Extracting features...")
examples = go.generate_training_data(filtered; progress_update=500, features=feats)
X = go.batch_training_examples([x[1] for x in examples])
Y = go.batch_training_examples([x[2][:] for x in examples])

println(STDERR, "Initializing model...")
clf = go.CLF_DCNN(feats)

println(STDERR, "Training...")
tm = time()
checkpoint_path = "../models/$(MODEL_NAME)_checkpoint.hf5"
go.save_model(clf, "../models/", MODEL_NAME, save_weights=false, save_yaml=true) 
go.train_model(clf, X, Y, epochs=epochs, validation_split=0.05, hf5path=checkpoint_path)
println("Took $(time() - tm) seconds")

println(STDERR, "Saving model...")
go.save_model(clf, "../models/", MODEL_NAME, save_weights=true, save_yaml=false) 
