# train_alphazero.py
import os
import time
from torch.nn.modules import loss

from Chessbasic import GameState
from AI import AlphaZeroAI, AlphaZeroNetwork
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    # 初始化游戏状态
    game_state = GameState()


    # 创建一个简单的神经网络模型实例
    model = AlphaZeroNetwork(board_size=8, action_size=game_state.Getvalidmove().__len__())
    model_path = 'alpha_zero_model.ckpt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # 确保 game_state 有所需的方法
    assert hasattr(game_state, 'reset') and callable(game_state.reset), "Game state must have a 'reset' method."

    # 创建 AlphaZeroAI 实例，传入棋盘状态实例
    alpha_zero_ai = AlphaZeroAI(
        game=game_state,
        model=model,
        lr=0.001,
        cuda=False,
        num_simulations=10
    )

    num_episodes = 10  # 设置训练的回合数

    num_epochs = 2  # 设置训练的轮数
    batch_size = 32  # 批量大小
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        # 自我对弈生成训练数据
        training_data = alpha_zero_ai.self_play()

        # 训练神经网络
        for epoch in range(num_epochs):
            # 初始化进度条
            start_time = time.time()
            epoch_loss = 0
            total_samples = len(training_data) // batch_size
            for batch_index, (state_batch, policy_batch, value_batch) in enumerate(
                    alpha_zero_ai.train(training_data, epochs=num_epochs, batch_size=batch_size)):
                # 训练代码...
                epoch_loss += loss.item()

                # 计算当前的进度百分比
                current_time = time.time()
                elapsed_time = current_time - start_time
                average_time_per_sample = elapsed_time / (batch_index + 1) if (batch_index + 1) else 0
                estimated_total_time = average_time_per_sample * total_samples
                percentage_complete = (batch_index + 1) / total_samples * 100

                # 打印进度信息
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - {percentage_complete:.2f}% complete. Time elapsed: {elapsed_time:.2f}s. Estimated remaining time: {estimated_total_time:.2f}s")

            epoch_loss /= total_samples
            print(f"Epoch {epoch + 1}/{num_epochs} loss: {epoch_loss:.4f}")

        # 打印训练进度
        if episode % 1 == 0:
            print(f"Completed {episode} episodes")

    # 保存训练好的模型（如果需要）
    torch.save(alpha_zero_ai.model.state_dict(), 'alpha_zero_model.ckpt')


if __name__ == "__main__":
    main()