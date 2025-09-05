import random

grid_size = 3
start = (0, 0)
goal = (2, 2)
actions = ["yukarı", "aşağı", "sol", "sağ"]

def step(state, action):
    x, y = state
    if action == "yukarı": 
        x -= 1
    elif action == "aşağı": 
        x += 1
    elif action == "sol": 
        y -= 1
    elif action == "sağ": 
        y += 1

    if x < 0 or y < 0 or x >= grid_size or y >= grid_size:
        return state, -1, False
    if (x, y) == goal:
        return (x, y), 10, True
    return (x, y), 0, False #-> konum x ve y ödül 0,

state = start # 0,0
for i in range(10):
    action = random.choice(actions)
    new_state, reward, done = step(state, action)
    print(f"agent {action} gitti → {new_state} → ödül {reward}")
    state = new_state
    if done:
        print("Hedefe ulaştı!")
        break 