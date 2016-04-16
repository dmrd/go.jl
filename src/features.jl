## Feature extraction
# All features calculated relative to current player to move

function one_hot(counts::Array{Int, 2}; maxval=8)
    out = BitArray(N, N, maxval)
    for i in 1:(maxval-1)
        out[:, :, i] = counts == i
    end
    out[:, :, maxval] = counts .>= maxval
    out
end

#
function stone_color(board::Board)
    result = BitArray(N, N, 3)
    cp = current_player(board)
    result[:, :, 1] = board.board .== cp
    result[:, :, 2] = board.board .== -cp
    result[:, :, 3] = board.board .== EMPTY
    result
end

function ones(board::Board)
    fill!(BitArray(N, N), true)
end

function zeros(board::Board)
    fill!(BitArray(N, N), false)
end

function turns_since(board::Board; maxval=8)
    one_hot(board.cmove - board.order, maxval=maxval)
end

function liberties(board::Board; maxval=8)
    liberties = Array{Int}(N,N)
    for x in 1:N
        for y in 1:N
            point = Point(y,x)
            group = board.groups[Point(y,x)]
            liberties[point] = length(group.liberties)
        end
    end
    one_hot(liberties, maxval=maxval)
end

# Liberties after move
# Capture size of move
# self atari size of move
function after_move_features(board::Board; maxval=8)
    legal = legal_moves(board)
    liberties = zeros(Int, N, N)
    groupsize = zeros(Int, N, N)
    capture_size = zeros(Int, N, N)
    color = current_player(board)
    for move in legal
        group = board.groups[move]
        (friends, foes, empties) = get_friend_foe(board, move, color)
        liberties[move] = length(union(group.liberties, [f.liberties for f in friends]...))
        groupsize[move] = 1 + sum([length(f.members) for f in friends])
        for foe in foes
            if length(foe.liberties) == 1
                capture_size[move] = length(foe.members)
            end
        end
    end
    cat(3, one_hot(liberties, maxval=maxval),
        one_hot(groupsize, maxval=maxval),
        one_hot(capture_size, maxval=maxval))
end

function ladder_capture(board::Board)
end

function ladder_escape(board::Board)
end

function sensibleness(board::Board)
end

function player_color(board::Board)
    is_black = current_player(board) == BLACK
    fill!(BitArray(N, N), is_black)
end

function get_features(board::Board, features::Vector{Function})
    processed = Vector{BitArray}()
    for feature in features
        @show feature
        push!(processed, feature(board))
    end
    return cat(3, processed...)
end

DEFAULT_FEATURES = [player_color, liberties]
ALL_FEATURES = [player_color, liberties, ones, zeros, turns_since, after_move_features, player_color]
