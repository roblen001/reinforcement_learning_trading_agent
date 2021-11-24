'''File which is ran to see the agent in action

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Added function descriptions.
'''
import pandas as pd
from env import EthereumEnv
from models import Random_games

df = pd.read_csv('ETHUSD.csv')

lookback_window_size = 10
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:]  # 30 days

train_env = EthereumEnv(
    train_df, lookback_window_size=lookback_window_size, debug_mode=True)
test_env = EthereumEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(train_env, train_episodes=10, training_batch_size=500)
