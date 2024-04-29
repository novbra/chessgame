# MCTS.py
import math
import random
import numpy as np  # 导入numpy模块


from Chessbasic import GameState

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state.copy() if game_state else None
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.win_score = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.game_state.Getvalidmove())

    def softmax(x):
        """Compute softmax values for each setting in x."""
        e_x = np.exp(x - np.max(x))  # 减去最大值以提高数值稳定性
        return e_x / e_x.sum()

    def select_child(self, temperature=0):
        if not self.children:  # 确保有子节点
            self.expand()  # 如果没有子节点，先展开

        if temperature > 0:
            # 使用 softmax 函数根据访问次数和胜率来选择子节点
            unnormalized_probabilities = [
                self.child_selection_score(child, temperature) for child in self.children
            ]
            probabilities = MCTSNode.softmax(unnormalized_probabilities)
            return np.random.choice(self.children, p=probabilities)

        else:
            # 在温度为0时，选择胜率最高的节点
            return max(self.children, key=lambda child: (child.win_score / max(1, child.visit_count)))
    @staticmethod
    def child_selection_score(child, temperature):
        visit_count = child.visit_count
        win_score = child.win_score
        # 确保不会除以零
        visit_count = max(visit_count, 1)
        return win_score / visit_count + math.sqrt(2 * math.log(temperature) / visit_count)

    def expand(self):
        valid_moves = self.game_state.Getvalidmove()
        for move in valid_moves:
            new_state = self.game_state.Piecemove(move)
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
            valid_moves = state.Getvalidmove()
            if not valid_moves:  # 检查列表是否为空
                # print(f"No valid moves from state {state}")
                return -1  # 没有合法走法
            # print(f"Valid moves: {valid_moves}")  # 打印合法走法
            move = random.choice(valid_moves)
            new_state = state.Piecemove(move)
            if new_state is None or new_state.is_game_over():
                # print(f"Game over or invalid state after move {move} from state {state}")
                return -1  # 如果走法导致游戏结束或状态无效，返回-1
            state = new_state  # 更新state为新的游戏状态
        return 1 if state.is_game_over() and state.IswTomove else -1

class MCTS:
    def __init__(self, game_state, num_simulations=10):

        self.root = MCTSNode(game_state)
        self.num_simulations = num_simulations

    def get_move(self):
        for _ in range(self.num_simulations):
            node = self.root
            # 使用温度参数选择子节点，随着模拟的深入，可以降低温度参数的值
            temperature = max(1, self.num_simulations - _)  # 随着模拟次数增加，降低温度

            while node.is_fully_expanded():
                node = node.select_child(temperature=temperature)

            if not node.is_fully_expanded():
                node.expand()


            result = node.simulate(limit=10)
            print(result)
            node.backpropagate(result)

            # 使用 lambda 函数和 numpy 的 sqrt 函数来计算每个子节点的选择分数
        best_move = max(self.root.children, key=lambda child: (child.visit_count / np.sqrt(child.visit_count + 1))).move

        return best_move

        # best_move = max(self.root.children, key=lambda child: child.visit_count).move


