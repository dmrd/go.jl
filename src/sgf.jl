# Simple tool for incrementally reading basic SGF files

type SGF
    parts::Vector{AbstractString}
    cindex::Int
    finished::Bool
    error::Bool  # If we encountered an error in parsing
    cplayer::Color  # Track here so we can read without updating board state
end


# Returns nothing if the game contains unsupported features
function SGF(filename::AbstractString)
    f = open(filename, "r")
    lines = readall(f)

    # Check if there is a handicap - we want to ignore these
    handicap_regex = r"A[W|B]\[(|..)\]"
    if match(handicap_regex, lines) != nothing
        println("contains handicaps: $(filename)")
        return nothing
    end

    moves = split(lines, ";")
    sgf = SGF(moves, 1, false, false, BLACK)
    sgf
end

# Utility function
# Use to filter out games between low ranked players
function players_over_rating(filename::AbstractString, minrating::Int)
    open(filename) do f
        lines = readall(f)
    end
    wrating_regex = r"WR\[[0-9]{4}\]"
    brating_regex = r"BR\[[0-9]{4}\]"
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

function find_sgf(folder::AbstractString)
    files = Vector{ByteString}
    folders = Vector{ByteString}
end


function get_next_move(sgf::SGF)
    regex = r"(W|B)\[(|..)\]"  # W[xx] / B[xx] / W[] / B[]
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
                move = Point(coords[1] - 'a' + 1, coords[2] - 'a' + 1)
            end
            sgf.cplayer = -sgf.cplayer
        end
    end
    move
end
