'''File which is ran to see the agent in action

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Reset the test data index to not have key issues.
'''
import pandas as pd
from env import EthereumEnv, CustomAgent
from models import Random_games, train_agent, test_agent
from tensorflow.keras.optimizers import Adam, RMSprop

df = pd.read_csv('ETHUSD.csv', index_col=False)
df = df.dropna().reset_index()

lookback_window_size = 50
test_window = 720  # 30 days
train_df = df[:-test_window-lookback_window_size]
test_df = df[-test_window-lookback_window_size:]

agent = CustomAgent(lookback_window_size=lookback_window_size,
                    lr=0.00001, epochs=1, optimizer=Adam, batch_size=32, model="Dense")
train_env = EthereumEnv(train_df, lookback_window_size=lookback_window_size)
train_agent(train_env, agent, visualize=False,
            train_episodes=20000, training_batch_size=500)
# Random_games(train_env, visualize=False,
#  train_episodes=10)
# test_env = EthereumEnv(
#     test_df, lookback_window_size=lookback_window_size)
# test_agent(test_env, agent, test_episodes=1,
#            folder="2021_12_04_09_55_Crypto_trader", name="1.41_Crypto_trader", comment="")

# agent = CustomAgent(lookback_window_size=lookback_window_size,
#                     lr=0.00001, epochs=1, optimizer=Adam, batch_size=32, model="CNN")
# test_env = EthereumEnv(
#     test_df, lookback_window_size=lookback_window_size, Show_reward=False)
# test_agent(test_env, agent, visualize=False, test_episodes=10,
#            folder="2021_01_11_23_48_Crypto_trader", name="1772.66_Crypto_trader", comment="")
# test_agent(test_env, agent, visualize=False, test_episodes=10,
#            folder="2021_01_11_23_48_Crypto_trader", name="1377.86_Crypto_trader", comment="")

# agent = CustomAgent(lookback_window_size=lookback_window_size,
#                     lr=0.00001, epochs=1, optimizer=Adam, batch_size=128, model="LSTM")
# test_env = EthereumEnv(
#     test_df, lookback_window_size=lookback_window_size, Show_reward=False)
# test_agent(test_env, agent, visualize=False, test_episodes=10,
#            folder="2021_01_11_23_43_Crypto_trader", name="1076.27_Crypto_trader", comment="")
