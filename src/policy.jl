using PyCall

@pyimport keras.models as models
@pyimport keras.layers.core as core
@pyimport keras.layers.convolutional as kconv
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
        model.load_weights(joinpath(folder, string(name, ".h5")))

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
Utility to handle difference between python and julia (column major) ordering
"""
function to_python(arr::AbstractArray)
    np.reshape(arr[:], reverse(size(arr)))
end

function to_python(arr::AbstractArray, size_for_python)
    np.reshape(arr[:], size_for_python)
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
      kconv.Convolution2D(64, 7, 7, activation="relu", border_mode="same", input_shape=input_shape),
      kconv.Convolution2D(48, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
      kconv.Convolution2D(48, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
      kconv.Convolution2D(32, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
      kconv.Convolution2D(32, 5, 5, activation="relu", border_mode="same", input_shape=input_shape),
      core.Flatten(),
      core.Dense(N*N, activation="softmax")
                                    ]),
      features)
end

function train_model(network::KerasNetwork, X, Y; epochs=20, batch_size=128, recompile=true, validation_split=0.25)
    # Reverse because julia <--> python
    X = to_python(X)
    Y = to_python(Y)
    if recompile
        println(STDERR, "Compiling model...")
        network.model.compile(loss="categorical_crossentropy",
                              optimizer="adadelta",
                              metrics=["accuracy"])
    end
    println(STDERR, "Fitting data...")
    network.model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size, verbose=2,
                      validation_split=validation_split)
end

function save_model(network::KerasNetwork, folder::AbstractString, name::AbstractString)
    hf5path = joinpath(folder, string(name, ".h5"))
    ymlpath = joinpath(folder, string(name, ".yml"))
    isfile(hf5path) && (println(STDERR, "File exists: $(hf5path)"); return)
    isfile(ymlpath) && (println(STDERR, "File exists: $(ymlpath)"); return)
    network.model.save_weights(hf5path)
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

# A policy takes a board and outputs a probability distribution over moves

function choose_move(board::Board, policy::KerasNetwork)
    X = get_features(board, features=policy.features) * 1.0
    X = to_python(X, (1, reverse(size(X))...))  # Pad it out so it is a batch of size 1
    # Have to convert to float before passing in (TODO - make this clearer)
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
