import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道数不同，使用1x1卷积调整维度
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out

# 定义残差网络
class ChessResNet(nn.Module):
    def __init__(self):
        super(ChessResNet, self).__init__()

        # 输入层
        self.conv = nn.Conv2d(8, 256, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        # 残差块
        self.residual_blocks = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )

        # 策略头
        self.policy_conv = nn.Conv2d(256, 16, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 8 * 8, 2086)

        # 价值头
        self.value_conv = nn.Conv2d(256, 8, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 输入层
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # 残差块
        x = self.residual_blocks(x)

        # 策略头
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        policy_output = self.policy_fc(p)
        policy_output = F.softmax(policy_output, dim=1)

        # 价值头
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = self.relu(v)
        value_output = self.value_fc2(v)
        value_output = torch.tanh(value_output)

        return policy_output, value_output

class PolicyValueNet(nn.Module):
    def __init__(self, model_file=None, use_gpu=True):
        super(PolicyValueNet, self).__init__()

        self.use_gpu = use_gpu
        self.l2_const = 2e-3  # L2 正则化参数

        # 创建策略价值网络模型
        self.policy_value_net = model1()  # 需要替换为你自定义的模型结构
        self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=0.001, weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).float()

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)

        act_probs = np.exp(log_act_probs.numpy())
        return act_probs, value.numpy()



  # 此处要修改！！！！！！board改成gamestate

    def policy_value_fn(self, GameState):
        self.policy_value_net.eval()

        # 获取合法动作列表
        _ , legal_positions = GameState.Getvalidmove
        current_state = np.ascontiguousarray(GameState.current_state().reshape(-1, 8, 8, 8)).astype('float32')
        current_state = torch.tensor(current_state).float()

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(current_state)

        act_probs = np.exp(log_act_probs.numpy().flatten())  # 只取出合法动作
        act_probs = zip(legal_positions, act_probs[legal_positions])  # 返回动作概率，以及状态价值
        return act_probs, value.numpy()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        state_batch = torch.tensor(state_batch).float()
        mcts_probs = torch.tensor(mcts_probs).float()
        winner_batch = torch.tensor(winner_batch).float()

        # 清零梯度
        self.optimizer.zero_grad()

        # 前向传播
        log_act_probs, value = self.policy_value_net(state_batch)
        value = value.view(-1)  # reshape value
        value_loss = F.mse_loss(value, winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))

        loss = value_loss + policy_loss

        # 反向传播及优化
        loss.backward()
        self.optimizer.step()

        # 计算策略的熵，仅用于评估模型
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))

        return loss.item(), entropy.item()

# 测试

model1 = ChessResNet()
test_data = torch.ones(8, 8, 8, 8)
x_act , x_val = model1(test_data)

print(x_act.shape)
print(x_val.shape)