using HDF5

function move_onehot(move::Point)
    board = BitArray(N, N)
    fill!(board, false)
    board[linearindex(move)] = true
    board
end

function generate_training_data(filename::AbstractString;
                                features::Vector{Function}=DEFAULT_FEATURES)
    sgf = SGF(filename)
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
            features = get_features(board)
            answer = move_onehot(move)
            push!(examples, (features, answer))
        end
        play_move(board, move)
    end
    return examples
end


function generate_training_data(filenames::Vector{AbstractString};
                                progress_update=100,
                                features::Vector{Function} = DEFAULT_FEATURES)
    examples = Vector{Tuple{BitArray, BitArray}}()
    tm = time()
    for (i, filename) in enumerate(filenames)
        if i % progress_update == 0
            println(STDERR, "$(i)/$(length(filenames)): $(time() - tm)")
        end
        push!(examples, generate_training_data(filename)...)
    end
    examples
end

function write_hdf5(filename::AbstractString, examples::Vector{Tuple{BitArray, BitArray}})
    data = hcat([x[1] for x in examples]...)
    label = hcat([x[2] for x in examples]...)
    h5open(filename, "w") do file
        # Convert to float and write
        write(file, "data", 1.0 * data)
        write(file, "label", 1.0 *label)
    end
end

# Utility function
# Use to filter out games between low ranked players
function players_over_rating(filename::AbstractString, minrating::Int)
    open(filename) do f
        lines = readall(f)
        wrating_regex = r"WR\[([0-9]{4})\]"
        brating_regex = r"BR\[([0-9]{4})\]"
        mw = match(wrating_regex, lines)
        mb = match(brating_regex, lines)
        if mw == nothing || mb == nothing
            return false
        end
        if parse(Int, mw.captures[1]) < minrating || parse(Int, mb.captures[1]) < minrating
            return false
        end
        return true
    end
end

# find sgf files in a folder
function find_sgf(folder::AbstractString)
    files = Vector{AbstractString}()
    folders = Vector{AbstractString}()
    push!(folders, folder)
    while length(folders) > 0
        cpath = pop!(folders)
        contents = readdir(cpath)
        for entry in contents
            path = joinpath(cpath, entry)
            if isdir(path)
                push!(folders, path)
            elseif endswith(path, ".sgf")
                push!(files, path)
            end
        end
    end
    files
end


function generate_and_save_examples(folder::AbstractString, out_hf5_path::AbstractString;
                                    features::Vector{Function} = DEFAULT_FEATURES)
    files = find_sgf(folder)
    filtered_files =  filter(x -> players_over_rating(x, 2000), files)
    examples = generate_training_data(filtered_files)
    write_hdf5(out_hf5_path, examples)
end
