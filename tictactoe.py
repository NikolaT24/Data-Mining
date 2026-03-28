import sys
import math

def print_board(board):
    print("+---+---+---+")
    for row in board:
        print("| " + " | ".join(row) + " |")
        print("+---+---+---+")

def check_winner(board):
    for i in range(3):
        if board[i][0] != '_' and board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
        if board[0][i] != '_' and board[0][i] == board[1][i] == board[2][i]:
            return board[0][i]
    if board[0][0] != '_' and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] != '_' and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    return None

def is_full(board):
    return all(c != '_' for row in board for c in row)

def minimax(board, is_max, alpha, beta, depth=0):
    winner = check_winner(board)
    if winner == 'X':
        return 10 - depth
    elif winner == 'O':
        return depth - 10
    elif is_full(board):
        return 0

    if is_max:
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == '_':
                    board[i][j] = 'X'
                    score = minimax(board, False, alpha, beta, depth + 1)
                    board[i][j] = '_'
                    best = max(best, score)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        return best
        return best
    else:
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == '_':
                    board[i][j] = 'O'
                    score = minimax(board, True, alpha, beta, depth + 1)
                    board[i][j] = '_'
                    best = min(best, score)
                    beta = min(beta, score)
                    if beta <= alpha:
                        return best
        return best

def best_move(board, turn):
    best_val = -math.inf if turn == 'X' else math.inf
    mv = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == '_':
                board[i][j] = turn
                value = minimax(board, turn == 'O', -math.inf, math.inf)
                board[i][j] = '_'
                if turn == 'X' and value > best_val:
                    best_val = value
                    mv = (i, j)
                elif turn == 'O' and value < best_val:
                    best_val = value
                    mv = (i, j)
    return mv

def mode_judge():
    turn = input().strip().split()[1]
    board = []
    for _ in range(7):
        line = input().strip()
        if '|' in line:
            row = [c for c in line if c in ['X', 'O', '_']]
            board.append(row)
    if check_winner(board) or is_full(board):
        print(-1)
        return
    mv = best_move(board, turn)
    if mv is None:
        print(-1)
    else:
        print(mv[0] + 1, mv[1] + 1)

def mode_game():
    first = input().strip().split()[1]
    human = input().strip().split()[1]
    board = []
    for _ in range(7):
        line = input().strip()
        if '|' in line:
            row = [c for c in line if c in ['X', 'O', '_']]
            board.append(row)
    turn = first
    print_board(board)
    while True:
        if check_winner(board) or is_full(board):
            break
        if turn == human:
            try:
                r, c = map(int, input().split())
            except:
                continue
            if 1 <= r <= 3 and 1 <= c <= 3 and board[r-1][c-1] == '_':
                board[r-1][c-1] = turn
            else:
                continue
        else:
            mv = best_move(board, turn)
            if mv:
                board[mv[0]][mv[1]] = turn
                print(f"Agent plays: {mv[0]+1} {mv[1]+1}")
        print_board(board)
        turn = 'O' if turn == 'X' else 'X'
    w = check_winner(board)
    if w:
        print("WINNER:", w)
    else:
        print("DRAW")

def main():
    mode = input().strip()
    if mode == "JUDGE":
        mode_judge()
    elif mode == "GAME":
        mode_game()
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
