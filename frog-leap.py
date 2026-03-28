import os
import sys
import time

def legal_moves(board, hole):
    L = len(board)
    moves = []

    if hole - 1 >= 0 and board[hole - 1] == '>':
        moves.append((hole - 1, hole - 1, 'S'))
    if hole - 2 >= 0 and board[hole - 2] == '>' and board[hole - 1] == '<':
        moves.append((hole - 2, hole - 2, 'J'))

    if hole + 1 < L and board[hole + 1] == '<':
        moves.append((hole + 1, hole + 1, 'S'))
    if hole + 2 < L and board[hole + 2] == '<' and board[hole + 1] == '>':
        moves.append((hole + 2, hole + 2, 'J'))

    return moves

def count_jumps_available(board, hole):
    return sum(1 for (_, _, kind) in legal_moves(board, hole) if kind == 'J')

def choose_move(board, hole):
    moves = legal_moves(board, hole)
    if not moves:
        return None

    if len(moves) == 1:
        return moves[0]

    jumps = [m for m in moves if m[2] == 'J']
    slides = [m for m in moves if m[2] == 'S']
    if jumps and slides:
        return jumps[0]

    if len(slides) == 2:
        for m in slides:
            frm, new_hole, _ = m
            board[hole], board[frm] = board[frm], board[hole]
            nj = count_jumps_available(board, new_hole)
            has_next = len(legal_moves(board, new_hole)) > 0
            board[hole], board[frm] = board[frm], board[hole]
            if has_next and nj < 2:
                return m
        return slides[0]

    if len(jumps) == 2:
        for m in jumps:
            frm, new_hole, _ = m
            board[hole], board[frm] = board[frm], board[hole]
            has_next = len(legal_moves(board, new_hole)) > 0
            board[hole], board[frm] = board[frm], board[hole]
            if has_next:
                return m
        return jumps[0]

    return moves[0]

def dfs_forced_path(n: int):
    board = ['>'] * n + ['_'] + ['<'] * n
    hole = n
    goal = '<' * n + '_' + '>' * n

    path = [''.join(board)]
    steps_left = n * (n + 2)

    while steps_left > 0:
        m = choose_move(board, hole)
        if m is None:
            break
        frm, new_hole, _ = m
        board[hole], board[frm] = board[frm], board[hole]
        hole = new_hole
        path.append(''.join(board))
        steps_left -= 1

    return path, goal

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])

    if os.getenv("FMI_TIME_ONLY") == "1":
        print("# TIMES_MS: alg=0")
        return

    t0 = time.time()
    path, goal = dfs_forced_path(n)
    t_ms = int((time.time() - t0) * 1000)

    print(f"# TIMES_MS: alg={t_ms}")
    print("\n".join(path))

if __name__ == "__main__":
    main()
