#!/usr/local/bin/julia
using go

model_dir = joinpath(splitdir(@__FILE__)[1], "models")

nargs = length(ARGS)
if nargs == 0
    policy = go.RandomPolicy()
else
    if ARGS[1] == "random"
        policy = go.RandomPolicy()
    elseif ARGS[1] == "keras"
        if nargs != 2
            println(STDERR, "Must specify Keras network name to load")
            exit()
        end
        policy = go.KerasNetwork(model_dir, ARGS[2])
    else
        println(STDERR, "Unknown policy type $(ARGS[1])")
        exit()
    end
end

go.gtp(policy)
