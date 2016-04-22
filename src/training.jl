using HDF5

function move_onehot(move::Point)
    board = BitArray(N, N)
    fill!(board, false)
    board[linearindex(move)] = true
    board
end

"Generate training data of form [(features, move), ...] for a single sgf file"
function generate_training_data(filename::AbstractString;
                                features::Vector{Function}=DEFAULT_FEATURES)
    sgf = load_sgf(filename)
    examples = Vector{Tuple{BitArray, BitArray}}()
    if sgf == nothing
        return examples
    end
    board = Board()
    while true
        move = get_next_move(sgf)
        if move == nothing
            break
        end
        if move != PASS_MOVE
            computed_features = get_features(board, features=features)
            answer = move_onehot(move)
            push!(examples, (computed_features, answer))
        end
        play_move(board, move)
    end
    return examples
end

"Generate training data for many sgf files"
function generate_training_data(filenames::Vector{AbstractString};
                                progress_update=100,
                                features::Vector{Function} = DEFAULT_FEATURES)
    examples = Vector{Tuple{BitArray, BitArray}}()
    tm = time()
    for (i, filename) in enumerate(filenames)
        if i % progress_update == 0
            println(STDERR, "$(i)/$(length(filenames)): $(time() - tm)")
        end
	game = generate_training_data(filename, features=features)
        if length(game) > 0
            push!(examples, game...)
        end
    end
    examples
end

"Extract training data directly to an hdf5 file"
function extract_to_hdf5(hf5_filename::AbstractString, filenames::Vector{AbstractString}, features::Vector{Function}; progress_update=1000, nlabels = 81)
    example_data = get_features(Board(), features=features)
    data_dim = size(example_data)

    hf5 = h5open(hf5_filename, "w")
    data = d_create(hf5, "data", datatype(UInt8),
                 ((data_dim..., 1), (data_dim..., -1)),
                 "chunk", (data_dim..., 1))
    label = d_create(hf5, "label", datatype(UInt8),
                     ((nlabels, 1), (nlabels, -1)),
                     "chunk", (nlabels, 1))
    tm = time()
    nexamples = 0
    for (i, filename) in enumerate(filenames)
        if i % progress_update == 0
            println(STDERR, "$(i)/$(length(filenames)): $(time() - tm)")
        end
        game = generate_training_data(filename, features=features)
        if length(game) > 0
            new_N = nexamples + length(game)
            set_dims!(data, (data_dim..., new_N))
            set_dims!(label, (nlabels, new_N))
            data_batch = batch_training_examples([x[1] for x in game])
            label_batch = batch_training_examples([x[2][:] for x in game])
            # Assumes data is 3 dims per example and labels are 1
            data[:, :, :, nexamples+1:new_N] = data_batch
            label[:, nexamples+1:new_N] = label_batch
            nexamples = new_N
        end
    end
    close(hf5)
end

"""
Create a vector by appending the given arrays in memory.
Reshape such that they are concatenated along the last+1 dimension

Wrote this because
X = cat(4,[x[1] for x in examples]...);
is slow for some reason
"""
function batch_training_examples(arrs)
    sizeper = length(arrs[1])
    out = similar(arrs[1], sizeper * length(arrs))
    for (i, arr) in enumerate(arrs)
        start = sizeper * (i - 1) + 1
        finish = sizeper * (i)
        out[start:finish] = arrs[i][:]
    end
    reshape(out, size(arrs[1])..., length(arrs))
end
