import torch
import torch.nn as nn
import torch.optim as optim
from MCTS import MCTS  # 假设您有一个MCTS的实现
from Chessbasic import GameState  # 导入您提供的棋盘游戏状态类
import numpy as np
class AlphaZeroNetwork(nn.Module):
    def __init__(self, board_size, action_size):
        super(AlphaZeroNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * board_size * board_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size + 1)  # +1 是为了价值输出
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        policy, value = x[:, :-1], x[:, -1]
        return policy, value

class AlphaZeroAI:
    def __init__(self, game, model=None, lr=0.001, cuda=False, num_simulations=1000):
        self.game = game  # 棋盘游戏的实例，比如GameState
        self.model = model if model else AlphaZeroNetwork(game.board_size, game.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mcts = MCTS(self.game, self.model, self.device, num_simulations)  # 初始化MCTS

    def get_move(self):
        #使用MCTS选择一个走法
        return self.mcts.get_move()

    def train(self, examples, epochs=1, batch_size=32):
        #""“训练神经网络”""
        dataloader = torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for state, policy_targets, value_targets in dataloader:
                state = state.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)
                policy, value = self.model(state)
                policy_loss = nn.functional.mse_loss(policy, policy_targets)
                value_loss = nn.functional.mse_loss(value.view(-1), value_targets.view(-1))
                loss = policy_loss + value_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def self_play(self):
        #""“自我对弈过程，生成训练数据”""
        game_state = self.game.reset()
        while not game_state.is_terminal():
            move = self.get_move()
            game_state = game_state.step(move)
        return self.game.get_training_data()  # 假设您的游戏状态类有一个方法来获取训练数据



# class MCTSNode:
#     def __init__(self, state, parent=None, action=None):
#         self.state = state
#         self.parent = parent
#         self.children = {}
#         self.visit_count = 0
#         self.total_value = 0
#         self.untried_actions = get_possible_actions(state)
#         self.action = action
#     def select_child(self, temperature=1):
#         # 选择子节点，使用探索因子（温度）来平衡探索与利用
#         c = np.sqrt(self.visit_count)
#         values = [(child.visit_count / c) + np.exp(child.total_value / c) for child in self.children.values()]
#         return self.children[np.argmax(values)]
#     def expand(self, model):
#         action = self.untried_actions.pop()
#         new_state = get_next_state(self.state, action)
#         child = MCTSNode(new_state, self, action)
#         self.children[action] = child
#         return child
#     def backpropagate(self, value):
#         self.visit_count += 1
#         self.total_value += value
#         if self.parent:
#             self.parent.backpropagate(value)
# class AlphaZeroAI:
#     def __init__(self, board_size, action_size, model=None, lr=0.001, cuda=False):
#         self.model = model if model else AlphaZeroNetwork(board_size, action_size)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#         self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#     def get_move(self, board, num_simulations=1000):
#         root = MCTSNode(board.get_state())
#         for _ in range(num_simulations):
#             node = root
#             while node.untried_actions and not node.state.is_terminal:
#                 if len(node.children) < len(node.untried_actions):
#                     node = node.expand(self.model)
#                 else:
#                     node = node.select_child()
#             leaf_state = node.state
#             with torch.no_grad():
#                 policy, value = self.model(torch.Tensor(leaf_state).unsqueeze(0).to(self.device))
#             node.backpropagate(value.item())
#         best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
#         return best_action
#     def train(self, examples, epochs=1, batch_size=32):
#         dataloader = torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=True)
#         for epoch in range(epochs):
#             for state, policy_targets, value_targets in dataloader:
#                 state = state.to(self.device)
#                 policy_targets = policy_targets.to(self.device)
#                 value_targets = value_targets.to(self.device)
#                 policy, value = self.model(state)
#                 policy_loss = nn.functional.mse_loss(policy, policy_targets)
#                 value_loss = nn.functional.mse_loss(value.view(-1), value_targets.view(-1))
#                 loss = policy_loss + value_loss
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()