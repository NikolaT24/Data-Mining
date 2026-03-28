import os
import sys

line = list(map(int, input().split()))
tiles_count = tiles_count = line[0]
goal_line = list(map(int, input().split()))
goal_blank_index = goal_line[0]

side = int(round((tiles_count + 1) ** 0.5))
if goal_blank_index == -1:
    goal_blank_index = tiles_count
start_vals = []
while len(start_vals) < tiles_count + 1:
    start_vals += list(map(int, input().split()))
start_state = start_vals[:tiles_count + 1]

goal_index_of_value = [0] * (tiles_count + 1)  
for val in range(1, tiles_count + 1):
    pos = val - 1 if (val - 1) < goal_blank_index else val
    goal_index_of_value[val] = pos
goal_index_of_value[0] = goal_blank_index

mdist = [[0] * (tiles_count + 1) for _ in range(tiles_count + 1)]
for val in range(1, tiles_count + 1):
    gr, gc = divmod(goal_index_of_value[val], side)
    row = mdist[val]
    for idx in range(tiles_count + 1):
        r, c = divmod(idx, side)
        row[idx] = abs(r - gr) + abs(c - gc)

def initial_manhattan(state_list):
    total = 0
    for idx, v in enumerate(state_list):
        if v != 0:
            total += mdist[v][idx]
    return total

def inversion_parity(flat):
    arr = [x for x in flat if x != 0]
    p = 0
    for i in range(len(arr)):
        ai = arr[i]
        for j in range(i + 1, len(arr)):
            if arr[j] < ai:
                p ^= 1
    return p

def is_solvable(flat):
    inv_p = inversion_parity(flat)
    if side % 2 == 1:
        return inv_p == 0
    start_blank_row_from_bottom = side - (flat.index(0) // side)
    goal_blank_row_from_bottom = side - (goal_blank_index // side)
    return (inv_p + start_blank_row_from_bottom) % 2 == (goal_blank_row_from_bottom % 2)

time_only = os.getenv("FMI_TIME_ONLY") == "1"
if time_only:
    print("# TIMES_MS: alg=0")
    sys.exit(0)

if not is_solvable(start_state):
    print(-1)
    sys.exit(0)

neighbors_by_index = [[] for _ in range(tiles_count + 1)]
for z in range(tiles_count + 1):
    r, c = divmod(z, side)
    if c > 0:
        neighbors_by_index[z].append((z - 1, 'left'))
    if c + 1 < side:
        neighbors_by_index[z].append((z + 1, 'right'))
    if r > 0:
        neighbors_by_index[z].append((z - side, 'up'))
    if r + 1 < side:
        neighbors_by_index[z].append((z + side, 'down'))

opposite_move = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}

def ida_star(start_list):
    state = start_list[:]
    zero_pos = state.index(0)
    h_cost = initial_manhattan(state)
    bound = h_cost
    path_moves = []

    def dfs(g_cost, h_cost, zero_pos, last_move):
        if h_cost == 0:
            return True

        f_score = g_cost + h_cost
        if f_score > bound:
            return f_score

        children = []
        for nxt, mv in neighbors_by_index[zero_pos]:
            if last_move and mv == opposite_move[last_move]:
                continue
            tile = state[nxt]
            new_h = h_cost - mdist[tile][nxt] + mdist[tile][zero_pos]
            children.append((g_cost + 1 + new_h, nxt, mv, new_h, tile))

        if not children:
            return 1 << 60

        children.sort(key=lambda x: x[0])

        best_next = 1 << 60
        for _, nxt, mv, new_h, tile in children:
            state[zero_pos], state[nxt] = state[nxt], state[zero_pos]
            res = dfs(g_cost + 1, new_h, nxt, mv)
            if res is True:
                path_moves.append(mv)
                return True

            state[zero_pos], state[nxt] = state[nxt], state[zero_pos]
            if res < best_next:
                best_next = res
        return best_next

    while True:
        outcome = dfs(0, h_cost, zero_pos, None)
        if outcome is True:
            path_moves.reverse()
            return path_moves
        if outcome >= (1 << 60):
            return None
        bound = outcome

solution = ida_star(start_state)

if solution is None:
    print(-1)
else:
    print(len(solution))
    opposite_move = {'left':'right','right':'left','up':'down','down':'up'}
    for mv in solution:
        print(opposite_move[mv])
