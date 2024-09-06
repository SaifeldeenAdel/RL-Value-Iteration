import numpy as np

DISCOUNT = 0.99
R = 3
ROWS = 3
COLS = 3
ACTIONS = ["T", "B", "L", "R"]

rewards = np.array([R, -1, 10, -1, -1, -1, -1, -1, -1, -1])
transitions = np.zeros((9, 4, 9))
# transitions holds probability of transitions from a state to another by doing a certain action
# first parameter is s, second is the action taken, and third is s' aka the new position
# from any state s, you can take one of 4 actions, EACH of which would lead to possibly 3 different states
# if in main direction then 0.8, left or right, 0.1

action_map = {
    "T": -COLS,
    "B": COLS,
    "L": -1,
    "R": 1
}

prob_main = 0.8
prob_left = 0.1
prob_right = 0.1

# Define left and right action rotations
left_rotation = {"T": "L", "B": "R", "L": "B", "R": "T"}
right_rotation = {"T": "R", "B": "L", "L": "T", "R": "B"}


def initialise_transitions():
    # Fill the transition matrix
    for state in range(ROWS * COLS):  # for each state
        for action_index, action in enumerate(ACTIONS):  # for each of the four possible actions for the state
            main_action = action_map[action]
            left_action = action_map[left_rotation[action]]
            right_action = action_map[right_rotation[action]]

            row, col = divmod(state, COLS)

            # Calculate new positions for main, left, and right actions
            new_positions = {
                "main": state + main_action,
                "left": state + left_action,
                "right": state + right_action
            }

            for key, new_pos in new_positions.items():
                new_row, new_col = divmod(new_pos, COLS)
                if key == "main":
                    if (0 <= new_pos < ROWS * COLS) and ((action == "L" and col > 0) or
                                                         (action == "R" and col < COLS - 1) or
                                                         (action == "T" and row > 0) or
                                                         (action == "B" and row < ROWS - 1)):
                        transitions[state, action_index, new_pos] += prob_main
                    else:
                        transitions[state, action_index, state] += prob_main
                elif key == "left":
                    if (0 <= new_pos < ROWS * COLS) and ((left_rotation[action] == "L" and col > 0) or
                                                         (left_rotation[action] == "R" and col < COLS - 1) or
                                                         (left_rotation[action] == "T" and row > 0) or
                                                         (left_rotation[action] == "B" and row < ROWS - 1)):
                        transitions[state, action_index, new_pos] += prob_left
                    else:
                        transitions[state, action_index, state] += prob_left
                elif key == "right":
                    if (0 <= new_pos < ROWS * COLS) and ((right_rotation[action] == "L" and col > 0) or
                                                         (right_rotation[action] == "R" and col < COLS - 1) or
                                                         (right_rotation[action] == "T" and row > 0) or
                                                         (right_rotation[action] == "B" and row < ROWS - 1)):
                        transitions[state, action_index, new_pos] += prob_right
                    else:
                        transitions[state, action_index, state] += prob_right


# ASK ABOUT WHAT IF WE GO OUT OF BOUNDS AND WE STILL NEED IT

def value_iteration():
    values = np.zeros(9)
    new_values = np.zeros(9)
    V_hist = [values.copy()]

    diff = 1

    while diff >= 1e-6:  # while difference is large enough
        diff = 0
        for state in range(9):  # for each of the 9 states
            q = np.zeros(4)  # initialize action values for state with 0s (we don't know their values yet)
            v_old = values[state]

            for action in range(4):  # calculate action value for each action
                q[action] = sum(
                    transitions[state, action, s] * (rewards[state] + DISCOUNT * values[s]) for s in range(9))
                # probability of transition from current state to state s by doing action
                # print(q)
            #print(f"state: {state}\n Q: {q}")
            new_values[state] = np.max(q)

            diff = max(diff, abs(v_old - new_values[state]))
        # update el values
        values = new_values.copy()
        #print(f"diff: {diff}")
        #print(f"iteration {i} :: values after iteration: {values}")
    return values


def policy_derivation(values):
    # taking in values
    # want to return policy it represents
    # for each state s
    # check the four directions possible
    # and take the max
    # and go for that direction
    # we will say policy is 9x4
    # row represents state you are in
    # column represents best ACTION to do
    policy = np.zeros((9, 4))
    for state in range(9):  # for each state
        q = np.zeros(4)
        for action in range(4):  # calculate action value for each action
            q[action] = sum(
                transitions[state, action, s] * (rewards[state] + DISCOUNT * values[s]) for s in range(9))
        max_action_index = np.argmax(q)
        policy[state, max_action_index] = 1
    return policy


def display_policy(policy):
    action_symbols = ["↑", "↓", "←", "→"]
    
    for i in range(ROWS):
        row_policy = ""
        for j in range(COLS):
            state = i * COLS + j
            action = np.argmax(policy[state])
            row_policy += f" {action_symbols[action]} "
        print(row_policy)


def main():
    initialise_transitions()
    print("transition matricies:\n")
    print(transitions)
    values = value_iteration()
    print(f"final values:\n{values}")
    policy = policy_derivation(values)
    print(f"final policy maticies:\n{policy}")
    print("final optimal policy:\n")
    display_policy(policy)
    # print(values)
    # print(transitions)


if __name__ == "__main__":
    main()
