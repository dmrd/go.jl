#!/usr/local/bin/julia
# Assumes agent goes first always for now
using OpenAIGym
using go

const debug = false

model_dir = joinpath(splitdir(@__FILE__)[1], "../models")

function usage()
    println(STDOUT, "Usage: ./openai_gym.jl exp_name rounds {random|keras [model name]}")
    exit()
end

nargs = length(ARGS)
exp_name = ARGS[1]
nrounds = parse(Int, ARGS[2])

if nargs != 3
    if ARGS[3] == "random"
        policy = go.RandomPolicy()
    elseif ARGS[3] == "keras"
        if nargs != 4
            usage()
        end
        policy = go.KerasNetwork(model_dir, ARGS[4])
    else
        usage()
    end
else
    usage()
end

function getempty(board)
    reshape(board[3, :, :], (go.N, go.N)).'
end

function get_opponent_move(last_empty, cur_empty, last_move::Int=0)
    # @show last_empty
    # @show cur_empty
    move = 0
    for i = 1:(go.N * go.N)
        if last_empty[i] == 1 && cur_empty[i] == 0 && i != last_move
            move = i
        end
    end
    if move == 0
        @show cur_empty
        @show last_empty
        @show last_move
        return go.PASS_MOVE
    else
        return from_gym_coords(go.pointindex(move))
    end
end

"""
gym indexes in row order from upper left of the board
 while we index from lower left along columns.
Together with from_gym_coords, transform between them.
"""
function to_gym_coords(p::go.Point)
    go.Point(p[2], go.N-p[1] + 1)
end

function from_gym_coords(p::go.Point)
    go.Point(go.N - p[2] + 1, p[1])
end

function run_episode()
    state = reset(env)
    localboard = go.Board()

    # TODO: The coordinate transforms are hacky and uncommented, but they work.
    last_empty = getempty(state.observation)
    opponent_move = go.EMPTY_MOVE # Init here so it stays in scope for after loop
    while !state.done
        # Our move
        if opponent_move == go.PASS_MOVE
            move = go.PASS_MOVE
            state = step(env, 81)
            break
        else
            move = go.choose_move(localboard, policy)
            state = step(env, go.linearindex(to_gym_coords(move)) - 1)
        end
        go.play_move(localboard, move)


        # Opponent move
        board = state.observation
        empty = getempty(board)
        opponent_move = get_opponent_move(last_empty, empty, go.linearindex(to_gym_coords(move)))

        if debug
            @show move
            @show opponent_move
        end

        go.play_move(localboard, opponent_move)
        last_empty = empty

        if debug
            println(localboard)
            display(env)
        end
        flush(STDOUT)
        flush(STDERR)
    end
    println(localboard)
    display(env)
    @show opponent_move
    @show state.reward
    @show go.calculate_score(localboard)
end

env = Env("Go$(go.N)x$(go.N)-v0")
env.env[:monitor][:start](exp_name)

for episode in 1:nrounds
    println("Episode $(episode)/$(nrounds)")
    run_episode()
end

env.env[:monitor][:close]()
