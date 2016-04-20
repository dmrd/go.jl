# Simple utils for incrementally reading basic SGF files

type SGF
    parts::Vector{AbstractString}
    cindex::Int
    finished::Bool
    error::Bool  # If we encountered an error in parsing
    cplayer::Color  # Track here so we can read without updating board state
end

# Example reading in SGF and updating board:
function playout_sgf(filecontents::AbstractString)
    sgf = go.SGF(filecontents)
    board = go.Board()
    while true
        move = get_next_move(sgf)
        if move == nothing
            break
        end
        play_move(board, move)
    end
    board
end

function load_sgf(filename::AbstractString)
    f = open(filename, "r")
    contents = readall(f)
    SGF(contents)
end

# Returns nothing if the game contains unsupported features
function SGF(contents::AbstractString; debug=false)
    # Check if there is a handicap - we want to ignore these
    handicap_regex = r"A[W|B]\[(|..)\]"
    if match(handicap_regex, contents) != nothing
        debug && println(STDERR, "Contains handicaps")
        return nothing
    end

    moves = split(contents, ";")
    sgf = SGF(moves, 1, false, false, BLACK)
    sgf
end

"find sgf files in a folder"
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


function get_next_move(sgf::SGF)
    # Assumes the move is always right after semicolon
    regex = r"^(W|B)\[(|..)\]"  # W[xx] / B[xx] / W[] / B[]
    move = nothing
    while move == nothing
        if sgf.cindex > length(sgf.parts)
            sgf.finished = true
            return nothing
        end
        part = sgf.parts[sgf.cindex]
        m = match(regex, part)
        sgf.cindex += 1
        if m == nothing
            # Try the next one if this move has no matches
            continue
        else
            color = m.captures[1] == "B" ? BLACK : WHITE
            coords = m.captures[2]
            if color != sgf.cplayer
                println(STDERR, "Incorrect color move in SGF: $(part)")
                sgf.error = true
                return
            end
            if coords == ""
                move = PASS_MOVE
            else
                move = Point(N - (coords[2] - 'a'), coords[1] - 'a' + 1)
            end
            sgf.cplayer = -sgf.cplayer
        end
    end
    move
end

"""
Utility function
Use to filter a list of sgf files to games where both players are above `minrating`
"""
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
