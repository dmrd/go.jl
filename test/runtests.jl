#TODO: Real tests =(
using go
using Base.Test

@test 1 == 1

# Very simple end-to-end sanity test
# Fits a linear model to examples from a single SGF file.
# Not going to be accurate, but tests simple 
examples = go.generate_training_data("./example.sgf")
X = go.batch_training_examples([x[1] for x in examples])
Y = go.batch_training_examples([x[2][:] for x in examples])
clf = go.CLF_LINEAR(go.LIBERTIES)
go.train_model(clf, X, Y, epochs=100, validation_split=0.5)
b = go.Board()
move = go.choose_move(b, clf)
