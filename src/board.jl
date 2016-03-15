import Base.getindex
import Base.setindex!
using DataStructures

# CONVENTION: Point is (y,x)
# TODO: Optimize size of integers

typealias Point Tuple{UInt8, UInt8}
Point{T <: Integer}(a::T, b::T) = Point((a, b))
linearindex(n::Integer, point::Point) = n * (point[2] - 1) + point[1]

typealias Color Int8
typealias BoardArray Array{Color, 2}
const EMPTY = Color(0)
const WHITE = Color(-1)
const BLACK = Color(1)
const PASS_MOVE = (0,0)


# TODO: PERFORMANCE Benchmark intsets versus Vector{Int}
type Group
    color::Color
    members::Vector{Point}
    liberties::Set{Point}
    Group(color::Color) = new(color, Vector{Int}(), Set{Point}())
end

type GroupSet
    n::UInt8  # Board size
    groups::Vector{Nullable{Group}}  # linear index -> group
end
GroupSet(n::Int) = GroupSet(UInt8(n), fill(Nullable{Group}(), n*n))

# Get the group for a given point
getindex(groups::GroupSet, point::Point) = groups.groups[linearindex(groups.n, point)]
getindex(groups::GroupSet, point::Integer) = groups.groups[point]

setindex!(groups::GroupSet, group::Group, point::Point) = groups[linearindex(groups.n, point)] = group
setindex!(groups::GroupSet, group::Group, point::Integer) = groups.groups[point] = group

remove_liberty(group::Group, point::Point) = setdiff!(group.liberties, point)
add_liberty(group::Group, point::Point) = push!(group.liberties, point)

type Board
    current_player::Color
    size::UInt8
    ko::Point  # 0,0 if none
    board::BoardArray
    groups::GroupSet
    captured::UInt32  # add for white, subtract for black
    Board(n::Integer) = new(BLACK, UInt8(n), (0,0), zeros(Color, n, n), GroupSet(n), 0)
end


function merge_groups(groupset::GroupSet, groups::Set{Group})
    # Find the largest group by # liberties
    largest = first(groups)
    msize = length(largest.liberties)
    nmembers = 0
    for group in groups
        gsize = length(group.liberties)
        if gsize > msize
            largest = group
            msize = gsize
        end
        nmembers += length(group.members)
    end


    # Grow members vector to avoid repeated resizing
    idx = length(largest.members) + 1
    resize!(largest.members, nmembers)
    
    # Merge other groups in
    for group in groups
        if group == largest
            continue
        end
        for member in group.members
            # Add member
            merged_members[idx] = member
            idx += 1
            # Update group lookup
            groups.groups[member] = largest
        end
        union!(largest.liberties, group.liberties)
    end

    # Ensure no members of the group are counted as a liberty
    setdiff!(largest.liberties, largest.members)

    largest
end

function remove_group(board::Board, group::Group)
    groups = board.groups
    for member in group.members
        board[member] = EMPTY
        groups[member] = Nullable{Group}()
        for neighbor in get_neighbors(board.size, member)
            ng = groups[neighbor]
            # Add the removed stone position as a liberty to any neighboring group
            if !isnull(ng) && ng != group
                add_liberty(ng, member)
            end
        end
    end
end

#function add_stone(board::Board, point::Point)
function add_stone(board::Board, point::Point)
    groups = board.groups
    color = board.current_player
    if !isnull(groups[point])
        @assert false && "Point already in a group"
    end
    board[point] = color
    neighbors = get_neighbors(groups.n, point)

    allies = Set{Group}()
    foes = Set{Group}()

    # Create new group
    group = Group(color)

    groups[point] = group

    push!(group.members, point)

    # Group neighbors by type
    for neighbor in neighbors
        maybe_ng = groups[neighbor]
        if isnull(maybe_ng)
            push!(group.liberties, neighbor)
            continue
        end
        ng = get(maybe_ng)
        if ng.color == color
            push!(ally, ng)
        elseif ng.color != color
            push!(foes, ng)
        end
    end

    # Merge allies together
    if length(allies) > 0
        push!(allies, group)
        group = merge_groups(groupset, allies)
    end

    for foe in foes
        remove_liberty(foe, point)
        if length(foe.libertie) == 0
            remove_group(foe)
        end
    end

    group
end

function onboard(n::Integer, point::Point)
    point[1] >= 1 && point[1] <= n &&
    point[2] >= 1 && point[2] <= n
end

function get_neighbors(n::Integer, point::Point)
    x = point[1]
    y = point[2]
    result = Vector{Point}()
    for neighbor in [(UInt8(x - 1), y), (UInt8(x + 1), y), (x, UInt8(y - 1)), (x, UInt8(y + 1))]
        onboard(n, neighbor) && push!(result, neighbor)
    end
    result
end

getindex(b::BoardArray, p::Point) = b[p...]
setindex!(b::BoardArray, c::Color, p::Point) = b[p...] = c

# TODO: Are multiple redirections optimized out?
# Same for GroupSet
getindex(b::Board, p::Point) = b.board[p]
setindex!(b::Board, c::Color, p::Point) = b.board[p] = c

# Is not a suicide if
#   a. Any neighbor is empty
#   b. Any neighbor group is same color and has > 1 libery
#   c. Any neighbor group is opponent color and only has 1 liberty (now captured)
function is_suicide(board::Board, point::Point)
    neighbors = get_neighbors(board.size, point)
    # Any neighbor is free
    for neighbor in neighbors
        if board[neighbor] == EMPTY
            return false
        end
    end

    for neighbor in neighbors
        if board[neighbor] == board.current_player && board.groups[neighbor].nliberties > 1
            # Connecting to group that has other liberties
            return false
        elseif board[neighbor] == -board.current_player && board.groups[neighbor].nliberties == 1
            # Captures opposing group
            return false
        end
    end
    return true
end

# 1. Unoccupied
# 2. Not illegal due to Ko rule
# 3. Is not a suicide
function is_legal(board::Board, point::Point)
    # Move must be on board
    if !onboard(board.size, point)
        println("a")
        return false
    end

    # Position must be empty
    if board[point] != EMPTY
        println("b")
        return false
    end

    # Cannot violate Ko rule
    if board.ko == point
        println("d")
        return false
    end

    # Finally, check if it is a suicide
    return !is_suicide(board, point)
end

function play_move(board::Board, point::Point)
    # Check legality
    if !is_legal(board, point)
        # Should probably make this not an assert
        @assert false && "Illegal move"
    end

    board.ko = PASS_MOVE

    add_stone(board, point)
    board.current_player = -board.current_player
    board
end


##################
# Representation #
##################A
const ascii = Dict(
                   EMPTY => '.',
                   BLACK => 'X',
                   WHITE => 'O'
                   )

function board_repr(board::Board)
    rows = Vector{ASCIIString}()
    mapper = char -> ascii[char]
    for row in 1:board.size
        row_repr = join(map(mapper, board.board[row, :]), "")
        push!(rows, row_repr)
    end
    join(rows, "\n")
end
