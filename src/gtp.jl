# Go Text Protocol
# Standard go interface


# Based on implementation from Michi
function gtp()
    known_commands = ["boardsize", "clear_board", "komi", "play", "genmove",
                      "final_score", "quit", "name", "version", "known_command",
                      "list_commands", "protocol_version", "tsdebug"]
end
