for key, new_pos in new_positions.items():  # for each of the three possible new states
    # Handle boundaries and invalid moves
    if key == "main" and 0 <= new_pos < ROWS * COLS:  # if main and new_pos is valid
        if (action == "L" and col > 0) or (action == "R" and col < COLS - 1) or (
                action == "T" and row > 0) or (
                action == "B" and row < ROWS - 1):  # why this if if we already checked new pos?
            transitions[state, action_index, new_pos] += prob_main
    elif key == "left" and 0 <= new_pos < ROWS * COLS:
        if (left_rotation[action] == "L" and col > 0) or (
                left_rotation[action] == "R" and col < COLS - 1) or (
                left_rotation[action] == "T" and row > 0) or (
                left_rotation[action] == "B" and row < ROWS - 1):
            transitions[state, action_index, new_pos] += prob_left
    elif key == "right" and 0 <= new_pos < ROWS * COLS:
        if (right_rotation[action] == "L" and col > 0) or (
                right_rotation[action] == "R" and col < COLS - 1) or (
                right_rotation[action] == "T" and row > 0) or (
                right_rotation[action] == "B" and row < ROWS - 1):
            transitions[state, action_index, new_pos] += prob_right