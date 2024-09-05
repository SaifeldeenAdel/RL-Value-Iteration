import numpy as np

DISCOUNT = 0.99
R = 100
ROWS = 3
COLS = 3
ACTIONS = ["T", "B", "L", "R"]

rewards = np.array([R, -1,10,-1,-1,-1,-1,-1,-1,-1])
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

# Define left and right action rotations
left_rotation = {"T": "L", "B": "R", "L": "B", "R": "T"}
right_rotation = {"T": "R", "B": "L", "L": "T", "R": "B"}



def initialise_transitions():
  # Fill the transition matrix
  for state in range(ROWS * COLS):
    for action_index, action in enumerate(ACTIONS):
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
          # Handle boundaries and invalid moves
          if key == "main" and 0 <= new_pos < ROWS * COLS:
              if (action == "L" and col > 0) or (action == "R" and col < COLS - 1) or (action == "T" and row > 0) or (action == "B" and row < ROWS - 1):
                  transitions[state, action_index, new_pos] += prob_main
          elif key == "left" and 0 <= new_pos < ROWS * COLS:
              if (left_rotation[action] == "L" and col > 0) or (left_rotation[action] == "R" and col < COLS - 1) or (left_rotation[action] == "T" and row > 0) or (left_rotation[action] == "B" and row < ROWS - 1):
                  transitions[state, action_index, new_pos] += prob_left
          elif key == "right" and 0 <= new_pos < ROWS * COLS:
              if (right_rotation[action] == "L" and col > 0) or (right_rotation[action] == "R" and col < COLS - 1) or (right_rotation[action] == "T" and row > 0) or (right_rotation[action] == "B" and row < ROWS - 1):
                  transitions[state, action_index, new_pos] += prob_right
  

def value_iteration():
  values = np.zeros(9)
  V_hist =[values.copy()] 
  
  diff = 1

  while diff >= 1e-6:
    diff = 0
    for state in range(9):
      q = np.zeros(4)
      v_old = values[state]

      for action in range(4):
        q[action] = sum(transitions[state, action, s] * (rewards[state] + DISCOUNT * values[s]) for s in range(4))
        # print(q)

      values[state] = np.max(q)
      diff = max(diff, abs(v_old-values[state]))
  return values

def policy_derivation(values):
  pass

def display_policy(policy):
  pass

def main():
  initialise_transitions()
  values = value_iteration()
  policy = policy_derivation(values)

  display_policy(policy)
  print(values)
  # print(transitions)

      

if __name__=="__main__":
  main()

