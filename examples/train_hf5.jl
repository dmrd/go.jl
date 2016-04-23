using go

if length(ARGS) < 4
    println(STDERR, "usage: train_hf5.jl input_hf5file MODEL_NAME num_epochs examples_per_epoch")
    exit()
end

INPUT = ARGS[1]
MODEL_NAME = ARGS[2]
EPOCHS = parse(Int, ARGS[3])
NUM_EXAMPLES = parse(Int, ARGS[4])

# Set the feature extractors
feats = go.LIBERTIES

println(STDERR, "Initializing model...")
clf = go.CLF_DCNN(feats)

println(STDERR, "Training...")
tm = time()
checkpoint_path = "../models/$(MODEL_NAME)_checkpoint.hf5"
# Save the model structure
go.save_model(clf, "../models/", MODEL_NAME, save_weights=false, save_yaml=true) 
go.train_model(clf, INPUT,
               epochs=EPOCHS,
               examples_per_epoch=500,
               n_validation_examples=500,
               checkpoint_path=checkpoint_path)
println("Took $(time() - tm) seconds")

println(STDERR, "Saving model...")
go.save_model(clf, "../models/", MODEL_NAME, save_weights=true, save_yaml=false) 
