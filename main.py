 import pygame
# some imports for game + RL stuff
import random
import numpy as np
import pickle
import os
from collections import defaultdict

# Game dimentions
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS  # each square size

# Colors lol
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)  # shows moves

# Pygame init stuuf
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shashki with Improved Q-Learning")

# the checker piece (draughts piece)
class Piece:
    PADDING = 10
    OUTLINE = 2

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.king = False
        self.x = 0
        self.y = 0
        self.calc_pos()

    def calc_pos(self):
        # figuring out x and y coords for drawing
        self.x = self.col * SQUARE_SIZE + SQUARE_SIZE // 2
        self.y = self.row * SQUARE_SIZE + SQUARE_SIZE // 2

    def make_king(self):
        # make it a king, woop
        self.king = True

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calc_pos()

    def draw(self, win):
        radius = SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(win, GREY, (self.x, self.y), radius + self.OUTLINE)
        pygame.draw.circle(win, self.color, (self.x, self.y), radius)
        if self.king:
            pygame.draw.circle(win, GREEN, (self.x, self.y), radius // 2)  # green dot = king

# the game board
class Board:
    def __init__(self):
        self.board = []
        self.red_left = self.white_left = 12  # total pieces at start
        self.red_kings = self.white_kings = 0
        self.create_board()

    def draw_squares(self, win):
        win.fill(BLACK)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(win, GREY, (col*SQUARE_SIZE, row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if row < 3 and (row + col) % 2 == 1:
                    self.board[row].append(Piece(row, col, RED))  # red starts at top
                elif row > 4 and (row + col) % 2 == 1:
                    self.board[row].append(Piece(row, col, WHITE))  # white at bottom
                else:
                    self.board[row].append(0)  # empty spot

    def draw(self, win):
        self.draw_squares(win)
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece != 0:
                    piece.draw(win)

    def get_piece(self, row, col):
        return self.board[row][col]

    def move(self, piece, row, col):
        # switch positions on board
        self.board[piece.row][piece.col], self.board[row][col] = 0, piece
        piece.move(row, col)

        # check if it became king
        if row == 0 and piece.color == WHITE:
            if not piece.king:
                piece.make_king()
                self.white_kings += 1
        elif row == ROWS - 1 and piece.color == RED:
            if not piece.king:
                piece.make_king()
                self.red_kings += 1

    def remove(self, pieces):
        # remove jumped pieces
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if piece.color == RED:
                self.red_left -= 1
            else:
                self.white_left -= 1

    def get_valid_moves(self, piece):
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row

        # check directions depending on color + king status
        if piece.color == RED or piece.king:
            moves.update(self._traverse_left(row + 1, min(row + 3, ROWS), 1, piece.color, left))
            moves.update(self._traverse_right(row + 1, min(row + 3, ROWS), 1, piece.color, right))

        if piece.color == WHITE or piece.king:
            moves.update(self._traverse_left(row - 1, max(row - 3, -1), -1, piece.color, left))
            moves.update(self._traverse_right(row - 1, max(row - 3, -1), -1, piece.color, right))

        return moves

    # helper fns for finding moves
    def _traverse_left(self, start, stop, step, color, left, skipped=[]):
        moves = {}
        last = []
        for r in range(start, stop, step):
            if left < 0:
                break

            current = self.board[r][left]
            if current == 0:
                if skipped and not last:
                    break  # invalid multi jump
                elif skipped:
                    moves[(r, left)] = last + skipped
                else:
                    moves[(r, left)] = last

                if last:
                    # go deeper if jump happened
                    row = max(r - 3, -1) if step == -1 else min(r + 3, ROWS)
                    moves.update(self._traverse_left(r + step, row, step, color, left - 1, skipped=last))
                    moves.update(self._traverse_right(r + step, row, step, color, left + 1, skipped=last))
                break
            elif current.color == color:
                break  # can't jump own guy
            else:
                last = [current]

            left -= 1

        return moves

    def _traverse_right(self, start, stop, step, color, right, skipped=[]):
        moves = {}
        last = []
        for r in range(start, stop, step):
            if right >= COLS:
                break

            current = self.board[r][right]
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, right)] = last + skipped
                else:
                    moves[(r, right)] = last

                if last:
                    row = max(r - 3, -1) if step == -1 else min(r + 3, ROWS)
                    moves.update(self._traverse_left(r + step, row, step, color, right - 1, skipped=last))
                    moves.update(self._traverse_right(r + step, row, step, color, right + 1, skipped=last))
                break
            elif current.color == color:
                break
            else:
                last = [current]

            right += 1

        return moves

    def get_all_pieces(self, color):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != 0 and piece.color == color:
                    pieces.append(piece)
        return pieces

    def evaluate(self):
        # simple score... could be smarter
        return self.white_left - self.red_left + (self.white_kings * 0.5 - self.red_kings * 0.5)

# our agent (q-learning brain)
class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.9995, min_epsilon=0.01):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def get_state_key(self, board):
        # flatten board into a tuple of numbers
        state = []
        for row in board.board:
            for piece in row:
                if piece == 0:
                    state.append(0)
                elif piece.color == RED:
                    state.append(1 if piece.king else 2)
                else:
                    state.append(-1 if piece.king else -2)
        return tuple(state)

    def choose_action(self, state, actions):
        # choose random or best move
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)

        q_values = []
        for action in actions:
            piece, move, skipped = action
            q_values.append(self.q_table.get((state, (piece.row, piece.col, move[0], move[1])), 0))

        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, next_actions):
        piece, move, skipped = action
        action_key = (piece.row, piece.col, move[0], move[1])
        current_q = self.q_table.get((state, action_key), 0)

        # best future q
        if next_actions:
            max_future_q = max(self.q_table.get((next_state, (p.row, p.col, m[0], m[1])), 0)
                               for p, m, s in next_actions)
        else:
            max_future_q = 0  # game over

        # update formula
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state, action_key)] = new_q

        # shrink eps slowly
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))

# helper to get all moves for a player
def get_all_moves(board, color):
    moves = []
    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        for move, skipped in valid_moves.items():
            moves.append((piece, move, skipped))
    return moves

# draws blue dots where moves can go
def draw_valid_moves(win, board, piece):
    valid_moves = board.get_valid_moves(piece)
    for move, _ in valid_moves.items():
        row, col = move
        pygame.draw.circle(win, BLUE,
                          (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                           row * SQUARE_SIZE + SQUARE_SIZE // 2),
                          15)

# main loop
def main():
    board = Board()
    agent = QAgent()
    run = True
    clock = pygame.time.Clock()
    selected = None
    turn = WHITE
    game_count = 0

    # try loading model
    agent.load_model('shashki_q_agent.pkl')

    while run:
        clock.tick(60)
        board.draw(WIN)
        if selected:
            draw_valid_moves(WIN, board, selected)
        pygame.display.update()

        # game over checks
        if board.red_left <= 0:
            print("White wins!")
            reward = 100
            if agent.last_state is not None:
                agent.update_q(agent.last_state, agent.last_action, reward, None, [])
            run = False
            continue
        elif board.white_left <= 0:
            print("Red wins!")
            reward = -100
            if agent.last_state is not None:
                agent.update_q(agent.last_state, agent.last_action, reward, None, [])
            run = False
            continue

        # agent (red) plays here
        if turn == RED:
            state = agent.get_state_key(board)
            actions = get_all_moves(board, RED)

            if not actions:
                print("White wins by blocking!")
                reward = 100
                if agent.last_state is not None:
                    agent.update_q(agent.last_state, agent.last_action, reward, None, [])
                run = False
                continue

            action = agent.choose_action(state, actions)
            piece, move, skipped = action

            agent.last_state = state
            agent.last_action = action

            board.move(piece, move[0], move[1])
            if skipped:
                board.remove(skipped)

            reward = board.evaluate()
            agent.last_reward = reward

            next_state = agent.get_state_key(board)
            next_actions = get_all_moves(board, WHITE)

            agent.update_q(state, action, reward, next_state, next_actions)
            turn = WHITE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if turn == WHITE:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = x // SQUARE_SIZE
                    row = y // SQUARE_SIZE

                    if selected:
                        moves = board.get_valid_moves(selected)
                        if (row, col) in moves:
                            board.move(selected, row, col)
                            skipped_pieces = moves[(row, col)]
                            if skipped_pieces:
                                board.remove(skipped_pieces)
                            selected = None
                            turn = RED
                        else:
                            selected = None
                            piece = board.get_piece(row, col)
                            if piece != 0 and piece.color == WHITE:
                                selected = piece
                    else:
                        piece = board.get_piece(row, col)
                        if piece != 0 and piece.color == WHITE:
                            selected = piece

    # save learned stuff
    agent.save_model('shashki_q_agent.pkl')
    pygame.quit()

if __name__ == '__main__':
    main()
