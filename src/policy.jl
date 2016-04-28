using PyCall

@pyimport keras.models as models
@pyimport yaml as pyyaml

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
# Training code in keras.jl
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

# A policy takes a board and outputs a probability distribution over moves

function choose_move(board::Board, policy::KerasNetwork)
    if board.last_move == PASS_MOVE
        return PASS_MOVE
    end
    moves = move_probs(board, policy)
    color = current_player(board)
    for move in moves
        point = pointindex(move)
        if is_legal(board, point, color)
            return point
        end
    end
    return PASS_MOVE
end

function move_probs(board::Board, policy::KerasNetwork)
    X = get_features(board, features=policy.features)
    X = to_python_array(X, size(X)..., 1) # Pad it out to create a batch
    probs = policy.model.predict(X)[:]
    sortperm(probs, rev=true)
end
