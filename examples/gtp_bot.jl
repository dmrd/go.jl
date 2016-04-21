#!/usr/local/bin/julia
using go

debug_log = joinpath(splitdir(@__FILE__)[1], "./gtp_debug.log")
model_dir = joinpath(splitdir(@__FILE__)[1], "../models")

f = open(debug_log, "w")  # stderr log
redirect_stderr(f)

function usage()
    println(STDOUT, "Usage: ./gtp_bot.jl {random|keras [model name]}")
    exit()
end

nargs = length(ARGS)
if nargs != 0
    if ARGS[1] == "random"
        policy = go.RandomPolicy()
    elseif ARGS[1] == "keras"
        if nargs != 2
            usage()
        end
        policy = go.KerasNetwork(model_dir, ARGS[2])
    else
        usage()
    end
else
    usage()
end

go.gtp(policy)
