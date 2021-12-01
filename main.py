'''File which is ran to see the agent in action

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Reset the test data index to not have key issues.
'''
import pandas as pd
from env import EthereumEnv
from models import Random_games, train_agent, test_agent

df = pd.read_csv('ETHUSD.csv', index_col=False)
df = df.dropna().reset_index()
lookback_window_size = 50
train_df = df[:-720-lookback_window_size]
test_df = df[-720-lookback_window_size:].reset_index()  # 30 days

train_env = EthereumEnv(
    train_df, lookback_window_size=lookback_window_size)
test_env = EthereumEnv(test_df, lookback_window_size=lookback_window_size)

# train_agent(train_env, visualize=False,
# train_episodes=20000, training_batch_size=500)
# # TODO: It doesnt make sense to have test episodes if the data is
# # not changing, the agent will keep taking the same actions
# # this is because it is timeseries data
test_agent(test_env, visualize=False, test_episodes=1)
# Random_games(train_env, visualize=False,
#              train_episodes=10)
