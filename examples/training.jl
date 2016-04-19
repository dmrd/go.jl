using go

if length(ARGS) != 1
    println(STDERR, "usage: training.jl MODEL_NAME num_epochs")
    exit()
end
MODEL_NAME = ARGS[1]
epochs = parse(Int, ARGS[2])

# Set the feature extractors
feats = go.LIBERTIES

println(STDERR, "Finding SGF files...")
files = go.find_sgf("../data/cgs/");

println(STDERR, "Finding SGF games with highly ranked players...")
filtered_files =  filter(x -> go.players_over_rating(x, 2200), files)

println(STDERR, "Extracting features...")
examples = go.generate_training_data(filtered_files; progress_update=500, features=feats)
X = go.batch_training_examples([x[1] for x in examples])
Y = go.batch_training_examples([x[2][:] for x in examples])

println(STDERR, "Initializing model...")
clf = go.CLF_DCNN(feats)
#clf = go.CLF_LINEAR(feats)

println(STDERR, "Training...")
tm = time()
go.train_model(clf, X, Y, epochs=epochs)
println("Took $(time() - tm) seconds")

println(STDERR, "Saving model...")
go.save_model(clf, "../models/", MODEL_NAME) 
