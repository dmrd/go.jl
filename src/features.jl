## Feature extraction
# All features calculated relative to current player to move

"""
Discretize a 2D integer matrix into `maxval` binary layers
Layers correspond to == 1, == 2, ... == (maxval - 1), >= maxval
"""
function discretize(counts::Array{Int, 2}; maxval=8)
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
    discretize(board.cmove - board.order, maxval=maxval)
end

function liberties(board::Board; maxval=3)
    liberties = Array{Int}(N,N)
    for x in 1:N
        for y in 1:N
            point = Point(y,x)
            group = board.groups[point]
            liberties[point] = length(group.liberties)
        end
    end
    discretize(liberties, maxval=maxval)
end

"Encode how many liberty counts broken down by stone color"
function perplayer_liberties(board::Board, maxval=3)
    cp = current_player(board)
    cur_liberties = Array{Int}(N,N)
    op_liberties = Array{Int}(N,N)
    fill!(cur_liberties, 0)
    fill!(op_liberties, 0)
    for x in 1:N
        for y in 1:N
            point = Point(y,x)
            group = board.groups[point]
            count = length(group.liberties)
            if board[point] == cp
                cur_liberties[point] = count
            else
                op_liberties[point] = count
            end
        end
    end
    cat(3, discretize(cur_liberties, maxval=3), discretize(op_liberties, maxval=3))
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
    cat(3, discretize(liberties, maxval=maxval),
        discretize(groupsize, maxval=maxval),
        discretize(capture_size, maxval=maxval))
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

#TODO: Provide features memory to write into to avoid unnecessary copies
function get_features(board::Board; features::Vector{Function}=DEFAULT_FEATURES)
    processed = Vector{BitArray}()
    for feature in features
        push!(processed, feature(board))
    end
    return cat(3, processed...)
end

# Run feature extraction on an empty board to get the dimensionality
function get_input_size(features::Vector{Function})
    size(get_features(Board(), features=features))
end

# TODO: Make a FeatureSet type to hold this
LIBERTIES = [perplayer_liberties]
DEFAULT_FEATURES = [stone_color, liberties]
ALL_FEATURES = [stone_color, liberties, ones, zeros, turns_since, after_move_features, player_color]

