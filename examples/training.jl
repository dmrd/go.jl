using go

feats = go.LIBERTIES

println(STDERR, "Finding SGF files")
#files = go.find_sgf("../data/cgs/");
files = go.find_sgf("../data/cgs/2015/12");
filtered_files =  filter(x -> go.players_over_rating(x, 2650), files)

println(STDERR, "Extracting features")
examples = go.generate_training_data(filtered_files; progress_update=500, features=feats);
X = go.batch_training_examples([x[1] for x in examples])
Y = go.batch_training_examples([x[2][:] for x in examples])

#clf = go.CLF_DCNN(feats)
clf = go.CLF_LINEAR(feats)
println(STDERR, "Training...")
tm = time()
go.train_model(clf, X, Y, epochs=1)
println("Took $(time() - tm) seconds")
go.save_model(clf, "../models/", "example_clf") 
