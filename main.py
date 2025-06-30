import pygame
import random
import pickle
import os
from collections import defaultdict

# Game dimensions
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CROWN = (255, 215, 0)  # Gold color for kings

# Initialize pygame
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers with Q-Learning")

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
        self.x = SQUARE_SIZE * self.col + SQUARE_SIZE // 2
        self.y = SQUARE_SIZE * self.row + SQUARE_SIZE // 2

    def make_king(self):
        self.king = True

    def move(self, row, col):
        self.row = row
        self.col = col
        self.calc_pos()

    def draw(self, win):
        radius = SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(win, self.color, (self.x, self.y), radius)
        if self.king:
            pygame.draw.circle(win, CROWN, (self.x, self.y), radius // 2)

class Board:
    def __init__(self):
        self.board = []
        self.red_left = self.white_left = 12
        self.red_kings = self.white_kings = 0
        self.create_board()

    def draw_squares(self, win):
        win.fill(BLACK)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(win, GREY, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if row < 3 and (row + col) % 2 == 1:
                    self.board[row].append(Piece(row, col, RED))
                elif row > 4 and (row + col) % 2 == 1:
                    self.board[row].append(Piece(row, col, WHITE))
                else:
                    self.board[row].append(0)

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
        self.board[piece.row][piece.col], self.board[row][col] = 0, piece
        piece.move(row, col)
        if row == 0 and piece.color == WHITE and not piece.king:
            piece.make_king()
            self.white_kings += 1
        elif row == ROWS - 1 and piece.color == RED and not piece.king:
            piece.make_king()
            self.red_kings += 1

    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = 0
            if piece.color == RED:
                self.red_left -= 1
                if piece.king:
                    self.red_kings -= 1
            else:
                self.white_left -= 1
                if piece.king:
                    self.white_kings -= 1

    def get_valid_moves(self, piece):
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row

        if piece.color == RED or piece.king:
            moves.update(self._traverse_left(row + 1, min(row + 3, ROWS), 1, piece.color, left))
            moves.update(self._traverse_right(row + 1, min(row + 3, ROWS), 1, piece.color, right))

        if piece.color == WHITE or piece.king:
            moves.update(self._traverse_left(row - 1, max(row - 3, -1), -1, piece.color, left))
            moves.update(self._traverse_right(row - 1, max(row - 3, -1), -1, piece.color, right))

        return moves

    def _traverse_left(self, start, stop, step, color, left, skipped=[]):
        moves = {}
        last = []
        for r in range(start, stop, step):
            if left < 0:
                break
            current = self.board[r][left]
            if current == 0:
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, left)] = last + skipped
                else:
                    moves[(r, left)] = last
                if last:
                    row = max(r - 3, -1) if step == -1 else min(r + 3, ROWS)
                    moves.update(self._traverse_left(r + step, row, step, color, left - 1, skipped=last))
                    moves.update(self._traverse_right(r + step, row, step, color, left + 1, skipped=last))
                break
            elif current.color == color:
                break
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
        return [piece for row in self.board for piece in row if piece != 0 and piece.color == color]

    def evaluate(self):
        piece_score = (self.white_left - self.red_left) * 1.0
        king_score = (self.white_kings - self.red_kings) * 1.5
        mobility_score = len(get_all_moves(self, WHITE)) - len(get_all_moves(self, RED))
        return piece_score + king_score + mobility_score * 0.1

class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.9995, min_epsilon=0.01):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.last_state = None
        self.last_action = None

    def get_state_key(self, board):
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
        if not actions:
            return None

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)

        q_values = []
        for action in actions:
            piece, move, skipped = action
            action_key = (piece.row, piece.col, move[0], move[1])
            q_values.append(self.q_table.get((state, action_key), 0))

        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, next_actions):
        if action is None:
            return

        piece, move, skipped = action
        action_key = (piece.row, piece.col, move[0], move[1])
        current_q = self.q_table.get((state, action_key), 0)

        if next_actions:
            max_future_q = max(
                self.q_table.get((next_state, (p.row, p.col, m[0], m[1])), 0)
                for p, m, _ in next_actions
            )
        else:
            max_future_q = 0

        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table[(state, action_key)] = new_q

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_model(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))

def get_all_moves(board, color):
    moves = []
    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        for move, skipped in valid_moves.items():
            moves.append((piece, move, skipped))
    return moves

class Game:
    def __init__(self, win):
        self.win = win
        self.board = Board()
        self.turn = WHITE
        self.valid_moves = {}
        self.selected = None
        self.ai = QAgent()
        self.ai.load_model("checkers_q_agent.pkl")
        self.game_over = False

    def update(self):
        self.board.draw(self.win)
        self.draw_valid_moves(self.valid_moves)
        pygame.display.update()

    def select(self, row, col):
        if self.selected:
            result = self._move(row, col)
            if not result:
                self.selected = None
                self.select(row, col)

        piece = self.board.get_piece(row, col)
        if piece != 0 and piece.color == self.turn:
            self.selected = piece
            self.valid_moves = self.board.get_valid_moves(piece)
            return True

        return False

    def _move(self, row, col):
        piece = self.board.get_piece(row, col)
        if self.selected and piece == 0 and (row, col) in self.valid_moves:
            self.board.move(self.selected, row, col)
            skipped = self.valid_moves[(row, col)]
            if skipped:
                self.board.remove(skipped)
            self.change_turn()
            return True
        return False

    def draw_valid_moves(self, moves):
        for move in moves:
            row, col = move
            pygame.draw.circle(self.win, BLUE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), 15)

    def change_turn(self):
        self.valid_moves = {}
        self.selected = None
        self.turn = RED if self.turn == WHITE else WHITE

        if self.board.red_left <= 0 or self.board.white_left <= 0 or \
           not get_all_moves(self.board, self.turn):
            self.game_over = True

    def ai_move(self):
        if self.turn == RED and not self.game_over:
            state = self.ai.get_state_key(self.board)
            actions = get_all_moves(self.board, RED)

            if actions:
                action = self.ai.choose_action(state, actions)
                if action:
                    piece, move, skipped = action
                    self.board.move(piece, move[0], move[1])
                    if skipped:
                        self.board.remove(skipped)
                    self.change_turn()
            else:
                self.game_over = True

def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Checkers')

    game = Game(win)
    clock = pygame.time.Clock()

    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                if game.turn == WHITE:
                    pos = pygame.mouse.get_pos()
                    row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE
                    game.select(row, col)

        if game.turn == RED and not game.game_over:
            game.ai_move()

        game.update()

        if game.game_over:
            if game.board.red_left <= 0:
                print("White wins!")
            elif game.board.white_left <= 0:
                print("Red wins!")
            else:
                print("No valid moves - game over!")
            pygame.time.delay(3000)
            return

if __name__ == "__main__":
    main()
