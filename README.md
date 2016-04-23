# go.jl

[![Build Status](https://travis-ci.org/dmrd/go.jl.svg?branch=master)](https://travis-ci.org/dmrd/go.jl)

This repository implements a Go bot framework end-to-end in Julia, including
functionality to maintain board state, extract features, read SGF files, and
communicate with other Go programs over GTP.

We use Keras over PyCall to train machine models to predict moves given a board state.  All the functionality is in place to implement more complex approaches, such as tree search or combining multiple policies together. Julia makes it practical to implement these speed intensive search operations.

# Using with Goban
A GTP interface is located at `examples/gtp_bot.jl`.  It is usable with [Goban](http://www.sente.ch/software/goban3/) on OSX by creating a GTP player with executable path `/path/to/go/examples/gtp_bot.jl`, and arguments `keras model_name` (models are read from the `models` directory).  Be sure that the correct board size `N` is set at the top of src/board.jl

# Getting data
The download script for data is located in `data/download.jl`.  Load it into the Julia repl and run `download_all(fetch_kgs(), "kgs")` to fetch the 19x19 kgs files, or `download_all(cgs_archives, "cgs") to download 9x9 cgs examples.

# Training a model
Example training script is located at `examples/training.jl`.  There are two main ways of training a model:

1. Extract examples from sgf files and start training a model immediately
  - e.g. `julia training.jl cgs_example_model --cgs --epochs 20`
2. Extract examples to an h5 file with `examples/extract.jl`, then train a model by reading from the h5 file.  Then is necessary to train large models, the data can take many GB (all 170k KGS games are 52gb in h5 format, for example).
  - `julia extract.jl ../data/cgs/ cgs2500.h5`
  - `julia training.jl cgs_example_model --h5path ../data/cgs2500.h5 --n_train 100000 --n_validation 5000`

Look in each script for more training options and settings.  The default model is a fairly large convolutional neural network which takes a long time to train. Model logic is located in `src/policy.jl`, and you can change the default clf for each training script (e.g. to `CLF_LINEAR` instead of `CLF_DCNN`, which will use a linear softmax instead of a many layer CNN).

Models are checkpointed whenever validation loss decreases by default.

# Installing


# Trying out on AWS


# Contributing

There's lots of places to go from here!  A few ideas:

- Optimizing the board representation and updates to make monte carlo playouts fast.
- A browser based interface instead of requiring installing Goban or similar
- Reinforcement learning