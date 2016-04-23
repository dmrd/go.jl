using PyCall

@pyimport keras.models as models
@pyimport keras.layers.core as core
@pyimport keras.layers.convolutional as kconv
@pyimport keras.callbacks as kcallbacks
@pyimport keras.utils.io_utils as k_io
@pyimport yaml as pyyaml
@pyimport numpy as np

abstract Policy
# Policy must have:
# choose_move(board::Board, policy::Policy)

type RandomPolicy <: Policy end

function choose_move(board::Board, policy::RandomPolicy)
    candidates = legal_moves(board)
    rand(candidates)
end

###
# Keras Policy
###

type KerasNetwork <: Policy
    # Ick - is there some way of improving typing here?
    model::Module
    features::Vector{Function}  # Feature extractors to run
    # Make it callable if given a raw object
    KerasNetwork(model::PyCall.PyObject, features) = new(pywrap(model), features)
    KerasNetwork(model::Module, features) = new(model, features)
end

"Load a Keras based policy from `folder` with `name`"
function KerasNetwork(folder::AbstractString, name::AbstractString)
    println(STDERR, "Loading Keras model from $(joinpath(folder, name))")
    open(joinpath(folder, string(name, ".yml"))) do file
        yaml = readall(file)
        model = pywrap(models.model_from_yaml(yaml))
        model.load_weights(joinpath(folder, string(name, ".hf5")))

        # Need to compile in order to predict anything even if we aren't training
        model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

        # Read in extra data on the feature maps this model was trained on
        # Do this with pyyaml because the yaml has !!python directives
        features = pyyaml.load(yaml)["features"]
        features = map(x -> eval(symbol(x)), features)

        println(STDERR, "Loaded model")
        KerasNetwork(model, features)
    end
end

"""
Convert array to row major, C style arrays.
Does not modify memory during conversion
"""
function to_python_array(arr::AbstractArray)
    to_python_array(arr, size(arr)...)
end

"""
Convert array to row major, C style arrays.
Does not modify buffer during conversion

Specifically for bit arrays -> UInt8

Specify dimensions IN JULIA COORDINATES
Given Julia array with dims: (1, 2, 3)
Return Python array with dims: (3, 2, 1)
"""
function to_python_array(arr::AbstractArray, dims...)
    reshaped = reshape(arr, dims)
    rounded = round(UInt8, reshaped)
    PyObject(rounded, true)
end

"Simple softmax classifier"
function CLF_LINEAR(features::Vector{Function})
    input_shape = get_input_size(features)
    KerasNetwork(models.Sequential([
                                    core.Flatten(input_shape=input_shape),
                                    core.Dense(N*N, input_dim=(N*N)),
                                    core.Activation("softmax")
                                    ]),
                 features)
end

"""
    Network from `Teaching Deep Convolutional Neural Networks to Play Go`
    Designed for 19x19, but should be fine for other sizes as well
    """
function CLF_DCNN(features::Vector{Function})
    # Reverse because julia <--> python
    input_shape = reverse(get_input_size(features))
    
    KerasNetwork(models.Sequential([
                                    kconv.Convolution2D(64, 7, 7, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(64, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(64, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(48, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(48, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(32, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    kconv.Convolution2D(32, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
                                    core.Flatten(),
                                    core.Dense(N*N, activation="softmax")
                                    ]),
                 features)
end

function compile(network::KerasNetwork)
    println(STDERR, "Compiling model...")
    network.model.compile(loss="categorical_crossentropy",
                          optimizer="adadelta",
                          metrics=["accuracy"])
end

function _callbacks(hf5path=nothing)
    callbacks = Vector()
    if hf5path != nothing
        checkpointer = kcallbacks.ModelCheckpoint(filepath=hf5path, verbose=1, save_best_only=true)
        push!(callbacks, checkpointer)
    end
    push!(callbacks, kcallbacks.ProgbarLogger())
    callbacks
end


# TODO: Make this less specific to this codebase / training case
"""
(UNFINISHED) Load an h5 file from within julia and feed the result in to `train_batch`

Mainly exists for "examples_per_epoch" flag. Should consider if there is a better work around (could store the games into smaller chunks in h5.  Then I can feed them into the general fit method.  This requires making sure the learning rate works fine.  Investigate internals.

TODO: async io

"""
function julia_train_h5(network::KerasNetwork, hdf5_path::AbstractString; epochs=20, examples_per_epoch=1000000, batch_size=32, recompile=true, n_validation_examples=50000, checkpoint_path=nothing)
    if recompile
        compile(network)
    end

    h5 = h5open(hdf5_path)
    X = h5["data"]
    Y = h5["label"]
    n_examples = size(X)[end]

    n_train_examples = n_examples - n_validation_examples
    start_tm = time()
    Xbatch = zeros(UInt8, size(X)[1:end-1]..., batch_size)
    Ybatch = zeros(UInt8, size(Y)[1:end-1]..., batch_size)

    train_start = 1 # iterating over dataset may take multiple epochs
    println(STDERR, "Training on $(n_train_examples), validate on $(n_validation_examples)")
    println(STDERR, "Examples per epoch: $(examples_per_epoch)")
    io = 0.0
    traintm = 0.0
    for epoch = 1:epochs
        examples_seen = 0
        println(STDERR, "Epoch $(epoch)/$(epochs):")
        train_loss = 0.0
        val_loss = 0.0

        batch_tm = time()
        while examples_seen < examples_per_epoch
            # May leave out a few examples at tail end for simplicity
            if train_start > n_train_examples - batch_size
                train_start = 1
                # TODO: Handle the wrap around case
            end
            batch_end = train_start + batch_size - 1
            # TODO: How to reduce IO overhead here?  Can I do it in a thread?
            tm = time()
            Xbatch[:, :, :, :] = X[:,:,:,train_start:batch_end]
            Ybatch[:, :] = Y[:, train_start:batch_end]
            io += time() - tm

            # TODO: May be able to just do this once since we reuse memory
            Xpy = to_python_array(Xbatch)
            Ypy = to_python_array(Ybatch)

            tm = time()
            train_loss += network.model.train_on_batch(Xpy, Ypy)
            traintm += time() - tm
            examples_seen += batch_size
        end

        for batch_offset = 1:batch_size:(n_validation_examples - batch_size)
            val_batch_start = n_train_examples + batch_offset + 1
            val_batch_end = val_batch_start + batch_size - 1
            Xbatch[:, :, :, :] = X[:,:,:,val_batch_start:val_batch_end]
            Ybatch[:, :] = Y[:, val_batch_start:val_batch_end]

            Xpy = to_python_array(Xbatch)
            Ypy = to_python_array(Ybatch)
            val_loss += network.model.test_on_batch(Xpy, Ypy)
        end
        println(STDERR, "Train Loss: $(train_loss) | Validation loss: $(val_loss) | Computed in $(time() - batch_tm) seconds")
        println(STDERR, "IO:$(io), Train:$(traintm)")
    end
end

" Train from an HDF5 file "
function keras_train_h5(network::KerasNetwork, hdf5_path::AbstractString; epochs=20, batch_size=32, recompile=true, n_validation=10000, n_train=nothing, checkpoint_path=nothing)
    if recompile
        compile(network)
    end

    #Open the file and see how many examples there are
    h5 = h5open(hdf5_path)
    X = h5["data"]
    Y = h5["label"]
    n_examples = size(X)[end]
    close(h5)

    if n_train == nothing
        n_train = n_examples - n_validation
    end

    # These act like matrices
    trainX = k_io.HDF5Matrix(hdf5_path, "data", 0, n_train)
    trainY = k_io.HDF5Matrix(hdf5_path, "label", 0, n_train)

    valX = k_io.HDF5Matrix(hdf5_path, "data", n_train, n_train + n_validation)
    valY = k_io.HDF5Matrix(hdf5_path, "label", n_train, n_train + n_validation)

    network.model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size, verbose=2,
                      validation_data=(valX, valY),
                      shuffle="batch", # Don't want to shuffle data too large for memory
                      callbacks=_callbacks(checkpoint_path))
end

function train_model(network::KerasNetwork, X, Y; epochs=20, batch_size=32, recompile=true, validation_split=0.25, checkpoint_path=nothing)
    if recompile
        compile(network)
    end

    # Handle the julia <--> python array conversion
    println(STDERR, "Doing python <--> Julia conversion...")
    X = to_python_array(X)
    Y = to_python_array(Y)
    println(STDERR, "Fitting model...")
    network.model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size, verbose=2,
                      validation_split=validation_split,
                      callbacks=_callbacks(checkpoint_path))
end

function save_model(network::KerasNetwork, folder::AbstractString, name::AbstractString; save_yaml=true, save_weights=true)
    if save_weights
        hf5path = joinpath(folder, string(name, ".hf5"))
        isfile(hf5path) && (println(STDERR, "File exists: $(hf5path)"); return)
        network.model.save_weights(hf5path)
    end

    if save_yaml
        ymlpath = joinpath(folder, string(name, ".yml"))
        isfile(ymlpath) && (println(STDERR, "File exists: $(ymlpath)"); return)
        yaml = network.model.to_yaml()

        # Add feature names
        # e.g. -> player_colors, liberties, ...
        # the split is to remove "go." - caused problems when reading in
        features = join(map(x -> split(string(x), ".")[end], network.features), ", ")
        yaml = string(yaml, "\nfeatures: [$(features)]")

        open(joinpath(folder, string(name, ".yml")), "w") do file
            write(file, yaml)
        end
    end
end

# A policy takes a board and outputs a probability distribution over moves

function choose_move(board::Board, policy::KerasNetwork)
    if board.last_move == PASS_MOVE
        return PASS_MOVE
    end
    X = get_features(board, features=policy.features)
    X = to_python_array(X, size(X)..., 1) # Pad it out to create a batch
    probs = policy.model.predict(X)[:]
    moves = sortperm(probs, rev=true)
    color = current_player(board)
    for move in moves
        point = pointindex(move)
        if is_legal(board, point, color)
            return point
        end
    end
    return PASS_MOVE
end
