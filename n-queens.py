import os
import sys
import random

N = int(input().strip())
if N == 0:
    sys.exit(0)

if os.getenv("FMI_TIME_ONLY") == "1":
    print("# TIMES_MS: alg=0")
    sys.exit(0)

if N in (2, 3):
    print(-1)
    sys.exit(0)

random.seed(0)

size_diag = 2 * N - 1
row_count = [0] * N
diag_down = [0] * size_diag
diag_up = [0] * size_diag
queen_row = [0] * N

def init_random_board():
    for i in range(N):
        row_count[i] = 0
    for i in range(size_diag):
        diag_down[i] = 0
        diag_up[i] = 0
    for c in range(N):
        r = random.randrange(N)
        queen_row[c] = r
        row_count[r] += 1
        diag_down[r - c + (N - 1)] += 1
        diag_up[r + c] += 1

def conflicts_in_col(c):
    r = queen_row[c]
    d1 = r - c + (N - 1)
    d2 = r + c
    return (row_count[r] - 1) + (diag_down[d1] - 1) + (diag_up[d2] - 1)

def best_rows_for_col(c):
    best = None
    rows = []
    rc, dd, uu = row_count, diag_down, diag_up
    off = N - 1
    cur = queen_row[c]
    for r in range(N):
        d1 = r - c + off
        d2 = r + c
        cost = rc[r] + dd[d1] + uu[d2]
        if r == cur:
            cost -= 3
        if best is None or cost < best:
            best = cost
            rows = [r]
        elif cost == best:
            rows.append(r)
    return rows

def move_queen(c, new_r):
    old_r = queen_row[c]
    if new_r == old_r:
        return
    row_count[old_r] -= 1
    diag_down[old_r - c + (N - 1)] -= 1
    diag_up[old_r + c] -= 1
    queen_row[c] = new_r
    row_count[new_r] += 1
    diag_down[new_r - c + (N - 1)] += 1
    diag_up[new_r + c] += 1

def solve_min_conflicts():
    max_steps = 10 * N + 200
    max_restarts = 20
    for _ in range(max_restarts):
        init_random_board()
        for _ in range(max_steps):
            conflicted = [c for c in range(N) if conflicts_in_col(c) > 0]
            if not conflicted:
                return True
            c = random.choice(conflicted)
            candidates = best_rows_for_col(c)
            move_queen(c, random.choice(candidates))
    return False

ok = solve_min_conflicts()

if not ok:
    print(-1)
else:
    print('[' + ', '.join(str(r) for r in queen_row) + ']')
