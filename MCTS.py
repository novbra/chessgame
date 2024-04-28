# MCTS.py
import random
import numpy as np  # 导入numpy模块
from Chessbasic import GameState

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.win_score = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game_state.Getvalidmove())

    def select_child(self, temperature=0):
        if temperature == 0:
            return max(self.children, key=lambda child: child.win_score / child.visit_count)
        else:
            return max(self.children, key=lambda child: self.child_selection_score(child, temperature))

    @staticmethod
    def child_selection_score(child, temperature):
        visit_count = child.visit_count
        win_score = child.win_score
        return win_score / visit_count + np.sqrt(2 * np.log(temperature) / visit_count)

    def expand(self):
        valid_moves = self.game_state.Getvalidmove()
        for move in valid_moves:
            new_state = self.game_state.step(move)
            new_node = MCTSNode(new_state, parent=self, move=move)
            self.children.append(new_node)

    def backpropagate(self, result):
        current = self
        while current:
            current.visit_count += 1
            current.win_score += result
            current = current.parent

    def simulate(self, limit):
        state = self.game_state
        for _ in range(limit):
            move = random.choice(state.Getvalidmove())
            state = state.step(move)
            if state.is_game_over():
                break
        return 1 if state.is_terminal() and state.IswTomove else -1

class MCTS:
    def __init__(self, game_state, num_simulations=1000):
        self.root = MCTSNode(game_state)
        self.num_simulations = num_simulations

    def get_move(self):
        for _ in range(self.num_simulations):
            node = self.root
            while node.is_fully_expanded():
                node = node.select_child(temperature=0)

            if not node.is_fully_expanded():
                node.expand()

            result = node.simulate(limit=10)
            node.backpropagate(result)

        best_move = max(self.root.children, key=lambda child: child.visit_count).move
        return best_move

