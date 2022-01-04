'''File which is ran to see the agent in action

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Reset the test data index to not have key issues.
'''
import pandas as pd
from pandas.core.frame import DataFrame
from env import EthereumEnv, CustomAgent
from models import Random_games, train_agent, test_agent
from multiprocessing_env import train_multiprocessing, test_multiprocessing
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn import preprocessing

if __name__ == "__main__":
    df = pd.read_csv('cryptoanalysis_data.csv', index_col=False)
    df = df.rename(columns={'price': 'Close'})
    df = df.rename(columns={'date': 'Date'})
    # TODO: going to need to add better data normalization
    # can also use standarization
    Close = list(df['Close'])
    df = df.drop(['Close'], axis=1)
    Date = df['Date']
    df = df.drop(['Date'], axis=1)
    column_maxes = df.max()
    df_max = column_maxes.max()
    column_mins = df.min()
    df_min = column_mins.min()
    normalized_df = (df - df_min) / (df_max - df_min)
    normalized_df['Close'] = Close
    normalized_df['Date'] = Date
    df = normalized_df

    lookback_window_size = 50
    test_window = 500
    train_df = df[:-test_window-lookback_window_size]
    test_df = df[-test_window-lookback_window_size:]

    agent = CustomAgent(lookback_window_size=lookback_window_size,
                        lr=0.00001, epochs=10, optimizer=Adam, batch_size=64, model="CNN")
    train_env = EthereumEnv(
        train_df, lookback_window_size=lookback_window_size)
    train_agent(train_env, agent, visualize=False,
                train_episodes=400000, training_batch_size=500)

    # train_multiprocessing(train_env, agent, train_df,
    #                       num_worker=12, training_batch_size=500, visualize=False, EPISODES=2000)

    # test_multiprocessing(EthereumEnv, CustomAgent, test_df, test_df_nomalized, num_worker=16, visualize=True,
    #                      test_episodes=1000, folder="2021_02_21_17_54_Crypto_trader", name="3263.63_Crypto_trader", comment="3 months")

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
