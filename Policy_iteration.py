import numpy as np

DISCOUNT = 0.99
R = 3
ROWS = 3
COLS = 3
ACTIONS = ["T", "B", "L", "R"]

rewards = np.array([R, -1, 10, -1, -1, -1, -1, -1, -1, -1])
transitions = np.zeros((9, 4, 9))

action_map = {
    "T": -COLS,
    "B": COLS,
    "L": -1,
    "R": 1
}

prob_main = 0.8
prob_left = 0.1
prob_right = 0.1

left_rotation = {"T": "L", "B": "R", "L": "B", "R": "T"}
right_rotation = {"T": "R", "B": "L", "L": "T", "R": "B"}

def initialise_transitions():
    for state in range(ROWS * COLS):
        for action_index, action in enumerate(ACTIONS):
            main_action = action_map[action]
            left_action = action_map[left_rotation[action]]
            right_action = action_map[right_rotation[action]]

            row, col = divmod(state, COLS)

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


def policy_generation():
    policy = np.random.randint(0, 4, ROWS * COLS)
    return policy

def policy_evaluation(policy, V):
    threshold = 1e-10
    while True:
        Q = 0
        for state in range(ROWS * COLS):
            v = V[state]
            V[state] = sum([transitions[state, policy[state], new_state] * (rewards[new_state] + DISCOUNT * V[new_state]) for new_state in range(ROWS * COLS)])
            Q = max(Q, abs(v - V[state]))
        if Q < threshold:
            break
    return V

def policy_improvement(policy, V):
    policy_stable = True
    for state in range(ROWS * COLS):
        old_action = policy[state]
        policy[state] = np.argmax([sum([transitions[state, action, new_state] * (rewards[new_state] + DISCOUNT * V[new_state]) for new_state in range(ROWS * COLS)]) for action in range(4)])
        if old_action != policy[state]:
            policy_stable = False
    return policy, policy_stable

def policy_iteration():
    initialise_transitions()
    policy = policy_generation()
    V = np.zeros(ROWS * COLS)
    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(policy, V)
        policy, policy_stable = policy_improvement(policy, V)
    return policy

def display_policy(policy):
    action_symbols = ["↑", "↓", "←", "→"]
    for row in range(ROWS):
        for col in range(COLS):
            print(action_symbols[policy[row * COLS + col]], end=" ")
        print()

def main():
    policy = policy_iteration()
    display_policy(policy)

if __name__ == "__main__":
    main()