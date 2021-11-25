'''This is a custom cryptocurrency trading environment created with
    openAi gym.

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Added function descriptions.
        - Removed data cleaning, know assume this has been done before.
        - Added transaction cost to performing orders.
        - Reward function is now calculating the percent gain difference
            between the benchmark, the increase of eth price over time
            and the profit in percent from trading.
        - Added a debug mode to produce a historical order data txt file 
            to make sure the order history align with what is expected when using the bot.
        - Completly changing the graphing feature in the env. No longer renders during but only
            plots after testing. New candle plot function added.
        - Keeping track of episode rewards for performance visualization.
'''
import pandas as pd
import numpy as np
import random
from collections import deque
from gym import spaces
from utils import Write_to_file, TradingGraph


class EthereumEnv:
    """Custom Ethereum Environment that follows gym interface"""

    def __init__(self, df, initial_balance=1000, lookback_window_size=50, trading_fee=0.1, debug_mode=False):
        '''Initiating the parameters.

            - df: cleaned pandas dataframe with historical crypto data.
            - initial_balance: int of the starting balance to trade.
            - lookback_window_size: int of number of candles we want
                our agent to see. (the candle period, ie daily, hourly... depends on the data given)
            - trading_fee: the percent of fee payed on every order.
            - render_range: amount of candles to render in chart.
        '''
        super(EthereumEnv, self).__init__()
        # TODO: add assertion to check the data
        # Define action space and state size and other custom parameters
        self.df = df
        self.trading_fee = trading_fee
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.debug_mode = debug_mode
        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        # TODO: the 10 will be switch the the number of columns in the crypto_analysis dataset
        self.state_size = (self.lookback_window_size, 10)

        # spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.state_size, dtype=np.float32)

        # actions ([hold, buy, sell])
        self.action_space = spaces.Discrete(3)

    # Reset the state of the environment to an initial state

    def reset(self, env_steps_size=0):
        '''Reset the env to an initial state.

            - env_step_size: int changes the step size for training the data.
                An alternative to random initial offset.
        '''
        self.trades = deque()  # for visualization
        self.episode_reward = 0
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.episode_reward = 0
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(
                self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume']
                                        ])

        state = np.concatenate(
            (self.market_history, self.orders_history), axis=1)

        return state  # reward, done, info can't be included

    # Get the data points for the given current_step
    def _next_observation(self):
        '''Get the data points for the given current_step state.
        '''
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume']
                                    ])
        obs = np.concatenate(
            (self.market_history, self.orders_history), axis=1)
        return obs

    # Execute one time step within the environment
    def step(self, action):
        '''Execute a step in the env.

            - action: int of the action to take
        '''
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to close
        current_price = self.df.loc[self.current_step, 'Close']
        Date = self.df.loc[self.current_step, 'Date']  # for visualization
        High = self.df.loc[self.current_step, 'High']  # for visualization
        Low = self.df.loc[self.current_step, 'Low']  # for visualization

        if action == 0:  # Hold
            pass
        # Buy with 100% of current balance TODO: confirm the math for crypto bought
        # Agent will only buy if it has at least more then 10% of the initial money remaining
        elif action == 1 and self.balance > self.initial_balance/10:
            self.crypto_bought = (
                self.balance - (self.trading_fee * self.balance)) / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': Date, 'High': High, 'Low': Low,
                               'total': self.crypto_bought, 'type': "buy"})

        # Sell 100% of current crypto held TODO: confirm the math for balance
        elif action == 2 and self.crypto_held > 0:
            self.crypto_sold = self.crypto_held
            self.balance += (self.crypto_sold * current_price) - \
                ((self.crypto_sold * current_price) * self.trading_fee)
            self.crypto_held -= self.crypto_sold
            self.trades.append(
                {'Date': Date, 'High': High, 'Low': Low, 'total': self.crypto_sold, 'type': "sell"})

        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        if self.debug_mode:
            Write_to_file(Date, self.orders_history[-1])
        # Calculate reward
        # Reward is the diff between total trading profits in percent - total eth gains in percent
        # TODO: maked sure the reward functions is doing the proper calculations
        buy_and_hold_gains_percent = (
            self.df.loc[self.current_step, 'Close'] / self.df.loc[0, 'Close']) * 100
        profit_percent = ((self.net_worth - self.initial_balance) /
                          self.initial_balance) * 100
        reward = profit_percent - buy_and_hold_gains_percent
        self.episode_reward += reward
        # TODO: this feel useless
        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        # info = [self.trades, self.net_worth]

        return obs, reward, done

    # render environment
    def render(self, visualize=False):
        '''Renders plot and output while agent is running.

            - visualize: bool, if True then a graph visualization will appear.
        '''
        if visualize:
            Date = self.df.loc[self.current_step, 'Date']
            Open = self.df.loc[self.current_step, 'Open']
            Close = self.df.loc[self.current_step, 'Close']
            High = self.df.loc[self.current_step, 'High']
            Low = self.df.loc[self.current_step, 'Low']
            Volume = self.df.loc[self.current_step, 'Volume']

            # Render the environment to the screen
            self.visualization.render(
                Date, Open, High, Low, Close, Volume, self.net_worth, self.trades)
