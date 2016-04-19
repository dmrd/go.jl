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
        push!(examples, generate_training_data(filename, features=features)...)
    end
    examples
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

function generate_and_save_examples(folder::AbstractString, out_hf5_path::AbstractString;
                                    features::Vector{Function} = DEFAULT_FEATURES)
    files = find_sgf(folder)
    filtered_files =  filter(x -> players_over_rating(x, 2000), files)
    examples = generate_training_data(filtered_files, features=features)
    write_hdf5(out_hf5_path, examples)
end
