function play_out(board::Board, num_rounds::Integer)
    forced_pass = false
    for i = 1:num_rounds
        candidates = legal_moves(board)
        if length(candidates) == 1
            # Forced pass for both players
            if forced_pass
                return board
            end
            forced_pass = true
        else
            forced_pass = false
        end
        move = rand(candidates)
        play_move(board, move)
    end
    board
end
