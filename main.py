'''File which is run to see the agent in action.

    Modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    Author: Roberto Lentini
    Email: roberto.lentini@mail.utoronto.ca
'''

import pandas as pd
from pandas.core.frame import DataFrame
from env import EthereumEnv, CustomAgent
from models import Random_games, train_agent, test_agent
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn import preprocessing

if __name__ == "__main__":
    # Read the dataset
    df = pd.read_csv('cryptoanalysis_data.csv', index_col=False)
    
    # Rename columns for consistency
    df = df.rename(columns={'price': 'Close'})
    df = df.rename(columns={'date': 'Date'})
    
    # Extract the 'Close' column and remove it from the dataframe
    Close = list(df['Close'])
    df = df.drop(['Close'], axis=1)
    
    # Extract the 'Date' column and remove it from the dataframe
    Date = df['Date']
    df = df.drop(['Date'], axis=1)
    
    # Normalize the data
    column_maxes = df.max()
    df_max = column_maxes.max()
    column_mins = df.min()
    df_min = column_mins.min()
    normalized_df = (df - df_min) / (df_max - df_min)
    normalized_df['Close'] = Close
    normalized_df['Date'] = Date
    df = normalized_df

    # Define the lookback window size and test window size
    lookback_window_size = 50
    test_window = 500

    # Split the dataframe into training and testing sets
    train_df = df[:-test_window - lookback_window_size]
    test_df = df[-test_window - lookback_window_size:]

    # Create a custom trading agent
    agent = CustomAgent(lookback_window_size=lookback_window_size,
                        lr=0.00001, epochs=10, optimizer=Adam, batch_size=64, model="CNN")
    
     
    # train_env = EthereumEnv(
    #     train_df, lookback_window_size=lookback_window_size)
    # train_agent(train_env, agent, visualize=False,
    #             train_episodes=400000, training_batch_size=500)

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
 
    # Create a testing environment
    test_env = EthereumEnv(test_df, lookback_window_size=lookback_window_size)
    
    # Test the agent in the environment
    test_agent(test_env, agent, visualize=False, test_episodes=1,
               folder="2022_01_18_10_40_Crypto_trader", name="122580.55_Crypto_trader", comment="")
