# Go Text Protocol
# Standard go interface


# Based on implementation from Michi
function gtp(policy::Policy)
    known_commands = ["boardsize", "clear_board", "komi", "play", "genmove",
                      "final_score", "quit", "name", "version", "known_command",
                      "list_commands", "protocol_version", "tsdebug"]
    board = Board()
    while true
        println(STDERR, "start====")
        flush(STDERR)
        if eof(STDIN)
            return
        end
        line = readline()
        println(STDERR, line)
        println(STDERR, "========")
        flush(STDERR)
        if line == ""
            continue
        end
        command = split(lowercase(line))
        if match(r"\d+", command[1]) != nothing
            cmdid = command[1]
            command = command[2:end]
        else
            cmdid = ""
        end
        ret = ""
        if command[1] == "boardsize"
            if parse(Int, command[2]) != N
                warn("Warning: Trying to play on different board $(command[1]) != $(N)")
                ret = nothing
            end
        elseif command[1] == "clear_board"
            board = Board()
        elseif command[1] == "komi"
            # Set komi
            board.komi = parse(Float64, command[2])
        elseif command[1] == "play"
            if length(command) == 2
                move = parse_coord(command[2])
            else 
                move = parse_coord(command[3])
            end
            println(STDERR, command)
            println(STDERR, move)
            play_move(board, move)
            println(STDERR, "played move")
        elseif command[1] == "genmove"
            move = choose_move(board, policy)
            play_move(board, move)
            ret = str_coord(move)
            println(STDERR, ret)
            println(STDERR, move)
            # TODO: Resigning, passing
        elseif command[1] == "final_score"
            score = calculate_score(board)
            if score == 0
                ret = "0"
            elseif score > 0
                ret = @sprintf("B+%f", score)
            elseif score < 0
                ret = @sprintf("W+%f", -score)
            end
        elseif command[1] == "name"
            ret = "go.jl"
        elseif command[1] == "version"
            ret = "Go bot written in Julia"
        elseif command[1] == "tsdebug"
            print_pos(board, output=STDOUT)
        elseif command[1] == "ldebug"
            # Print liberties of each stone
            print_liberties(board, output=STDOUT)
        elseif command[1] == "list_commands"
            ret = join(known_commands, "\n")
        elseif command[1] == "known_command"
            ret = command[2] in known_commands ? "true" : "false"
        elseif command[1] == "protocol_version"
            ret = "2"
        elseif command[1] == "quit"
            @printf("=%s \n\n", cmdid)
            break
        else
            warn("Warning: Ignoring unknown command - $(line)")
            ret = nothing
        end
        print_pos(board)
        if ret != nothing
            out = @sprintf("=%s %s\n\n", cmdid, ret)
        else
            out = @sprintf("?%s ???\n\n", cmdid)
        end
        @printf("%s", out)
        println(STDERR, out)
        flush(STDOUT)
        flush(STDERR)
    end
end
