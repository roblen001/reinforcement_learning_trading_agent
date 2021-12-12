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
        - Simplifying the env file. Only keeping functions relevant to the env class in this file.
        - Adding information to trades list for visualization purposes
        - Added new data and removed previously used data, bot is now trading on fundementals
'''
from matplotlib.pyplot import axis
from numpy.core.arrayprint import _leading_trailing
import pandas as pd
import numpy as np
import random
from collections import deque
from gym import spaces
import copy

from six import reraise
from utils import Write_to_file
from models import Actor_Model, Critic_Model, Shared_Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorboardX import SummaryWriter
from datetime import datetime as dt
import os


class CustomAgent:
    '''Ethereum trading agent.
    '''

    def __init__(self, lookback_window_size=50, lr=0.00005, epochs=1, optimizer=Adam, batch_size=32, model=""):
        self.lookback_window_size = lookback_window_size
        self.model = model

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # folder to save models
        self.log_name = dt.now().strftime("%Y_%m_%d_%H_%M")+"_Crypto_trader"

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (lookback_window_size, 20)

        # Neural Networks part bellow
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create shared Actor-Critic network model
        self.Actor = self.Critic = Shared_Model(
            input_shape=self.state_size, action_space=self.action_space.shape[0], lr=self.lr, optimizer=self.optimizer, model=self.model)
        # Create Actor-Critic network model
        # self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        # self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)

    # create tensorboard writer
    def create_writer(self, initial_balance, normalize_value, train_episodes):
        '''Tensorboard writer.
        '''
        self.replay_count = 0
        self.writer = SummaryWriter('runs/'+self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.start_training_log(
            initial_balance, normalize_value, train_episodes)

    def start_training_log(self, initial_balance, normalize_value, train_episodes):
        # save training parameters to Parameters.txt file for future
        with open(self.log_name+"/Parameters.txt", "w") as params:
            current_date = dt.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training start: {current_date}\n")
            params.write(f"initial_balance: {initial_balance}\n")
            params.write(f"training episodes: {train_episodes}\n")
            params.write(
                f"lookback_window_size: {self.lookback_window_size}\n")
            params.write(f"lr: {self.lr}\n")
            params.write(f"epochs: {self.epochs}\n")
            params.write(f"batch size: {self.batch_size}\n")
            params.write(f"normalize_value: {normalize_value}\n")
            params.write(f"model: {self.model}\n")

    def end_training_log(self):
        with open(self.log_name+"/Parameters.txt", "a+") as params:
            current_date = dt.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training end: {current_date}\n")

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d,
                  nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)

        # Compute advantages
        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        plt.plot(target,'-')
        plt.plot(advantages,'.')
        ax=plt.gca()
        ax.grid(True)
        plt.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(
            states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(
            states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay',
                               np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay',
                               np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        self.Actor.Actor.save_weights(
            f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(
            f"{self.log_name}/{score}_{name}_Critic.h5")

        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = dt.now().strftime('%Y-%m-%d %H:%M:%S')
                log.write(
                    f"{current_time}, {args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}\n")

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(
            os.path.join(folder, f"{name}_Critic.h5"))


class EthereumEnv:
    """Custom Ethereum Environment that follows gym interface"""

    def __init__(self, df, initial_balance=1000, lookback_window_size=50, trading_fee=0.1, debug_mode=False, normalize_value=40000):
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
        # self.market_history = deque(maxlen=self.lookback_window_size)

        # Blockchain analysis data
        self.blockchain_data = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value

    def reset(self, env_steps_size=0):
        '''Reset the env to an initial state.

            - env_step_size: int changes the step size for training the data.
                An alternative to random initial offset.
        '''
        self.total_trade_profits = 0
        self.prev_episode_orders = 0  # track previous episode orders count
        self.punish_value = 0
        self.net_worth_lst = deque()  # for visualization
        self.trades = deque()  # for visualization
        self.episode_reward = 0
        self.reward_lst = deque()  # for visualization
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0
        self.number_of_purchases = 0
        self.number_of_holds = 0
        self.number_of_sales = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(
                self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            # issue: we have a dumbfucking index
            # TODO: this breaks when you run with test data
            if 'level_0' in self.df.columns:
                self.df = self.df.drop(['level_0'], axis=1).reset_index()
            else:
                self.df = self.df.reset_index()
            current_step = self.current_step - i

            self.orders_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            # self.market_history.append([self.df.loc[current_step, 'Open'],
            #                             self.df.loc[current_step, 'High'],
            #                             self.df.loc[current_step, 'Low'],
            #                             self.df.loc[current_step, 'Close'],
            #                             self.df.loc[current_step, 'Volume']
            #                             ])
            self.blockchain_data.append([self.df.loc[current_step, 'Close'],
                                        # self.df.loc[current_step,
                                         #             'receive_count'],
                                         # self.df.loc[current_step,
                                         #             'sent_count'],
                                         # self.df.loc[current_step, 'avg_fee'],
                                         # self.df.loc[current_step, 'blocksize'],
                                         # self.df.loc[current_step,
                                         #             'btchashrate'],
                                         # self.df.loc[current_step,
                                         #             'OIL'],
                                         # self.df.loc[current_step,
                                         #             'ecr20_transfers'],
                                         # self.df.loc[current_step, 'GOLD'],
                                         # self.df.loc[current_step, 'searches'],
                                         # self.df.loc[current_step, 'hashrate'],
                                         # self.df.loc[current_step,
                                         #             'marketcap'],
                                         # self.df.loc[current_step,
                                         #             'difficulty'],
                                         # self.df.loc[current_step, 's&p500'],
                                         # self.df.loc[current_step,
                                         #             'transactionfee'],
                                         #  self.df.loc[current_step,
                                         #              'transactions'],
                                         # self.df.loc[current_step,
                                         #             'tweet_count'],
                                         # self.df.loc[current_step,
                                         #             'unique_adresses'],
                                         # self.df.loc[current_step, 'VIX'],
                                         # self.df.loc[current_step, 'UVYX']
                                         ])

        # state = np.concatenate(
        #     (self.market_history, self.orders_history), axis=1)

        state = self.blockchain_data
        return state  # reward, done, info can't be included

    # Get the data points for the given current_step
    def _next_observation(self):
        '''Get the data points for the given current_step state.
        '''
        # self.market_history.append([self.df.loc[self.current_step, 'Open'],
        #                             self.df.loc[self.current_step, 'High'],
        #                             self.df.loc[self.current_step, 'Low'],
        #                             self.df.loc[self.current_step, 'Close'],
        #                             self.df.loc[self.current_step, 'Volume']
        #                             ])

        self.blockchain_data.append([self.df.loc[self.current_step, 'Close'],
                                    #  self.df.loc[self.current_step,
                                     #              'receive_count'],
                                     #  self.df.loc[self.current_step,
                                     #              'sent_count'],
                                     #  self.df.loc[self.current_step, 'avg_fee'],
                                     #  self.df.loc[self.current_step,
                                     #              'blocksize'],
                                     #  self.df.loc[self.current_step,
                                     #              'btchashrate'],
                                     #  self.df.loc[self.current_step,
                                     #              'OIL'],
                                     #  self.df.loc[self.current_step,
                                     #              'ecr20_transfers'],
                                     #  self.df.loc[self.current_step, 'GOLD'],
                                     #  self.df.loc[self.current_step,
                                     #              'searches'],
                                     #  self.df.loc[self.current_step,
                                     #              'hashrate'],
                                     #  self.df.loc[self.current_step,
                                     #              'marketcap'],
                                     #  self.df.loc[self.current_step,
                                     #              'difficulty'],
                                     #  self.df.loc[self.current_step, 's&p500'],
                                     #  self.df.loc[self.current_step,
                                     #              'transactionfee'],
                                     #  self.df.loc[self.current_step,
                                     #              'transactions'],
                                     #  self.df.loc[self.current_step,
                                     #              'tweet_count'],
                                     #  self.df.loc[self.current_step,
                                     #              'unique_adresses'],
                                     #  self.df.loc[self.current_step, 'VIX'],
                                     #  self.df.loc[self.current_step, 'UVYX']
                                     ])
        # obs = np.concatenate(
        #     (self.market_history, self.orders_history), axis=1)

        obs = self.blockchain_data

        return obs

    # Execute one time step within the environment
    def step(self, action):
        '''Execute a step in the env.

            - action: int of the action to take
        '''
        self.reward = 0
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # print('networth prreeeee trade=================')
        # print(self.net_worth)
        # Set the current price to close
        current_price = self.df.loc[self.current_step, 'Close']
        Date = self.df.loc[self.current_step, 'Date']  # for visualization
        if action == 0:  # Hold
            pass
        # Buy with 100% of current balance TODO: confirm the math for crypto bought
        # Agent will only buy if it has at least more then 10% of the initial money remaining
        elif action == 1 and self.balance > self.initial_balance*0.05:
            self.crypto_bought = (
                self.balance - (self.trading_fee * self.balance)) / current_price
            self.balance = 0
            self.crypto_held += self.crypto_bought
            self.episode_orders += 1
            # net_worth = self.balance + self.crypto_held * current_price
            # TODO: remove the current price from list high and low removed
            self.trades.append({'Date': Date, 'Close': current_price,
                               'total': self.crypto_bought, 'type': 'buy',
                                'Net_worth': 0, 'Reward': None, 'current_price': current_price})

        # Sell 100% of current crypto held TODO: confirm the math for balance
        elif action == 2 and self.crypto_held*current_price > self.initial_balance*0.05:
            self.crypto_sold = self.crypto_held
            self.balance += (self.crypto_sold * current_price) - \
                ((self.crypto_sold * current_price) * self.trading_fee)
            self.crypto_held = 0
            self.episode_orders += 1
            # print('BUY 2 ====')
            # print(self.trades[-1]['total']*self.trades[-1]['Close'])
            # print('SELL==========')
            # print(self.balance)
            profits = self.balance - \
                self.trades[-1]['total']*self.trades[-1]['Close']
            # net_worth = self.balance + self.crypto_held * current_price
            # TODO: remove the current price from list removed 'High': High, 'Low': Low,
            self.trades.append({'Date': Date, 'Close': current_price,
                               'total': self.crypto_sold, 'type': 'sell',
                                'Net_worth': 0, 'Reward': None, 'current_price': current_price, 'profits': profits})

        self.net_worth = self.balance + (self.crypto_held * current_price)
        # print('networth post trade=================')
        # print(self.net_worth)
        self.orders_history.append(
            [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        if self.debug_mode:
            Write_to_file(Date, self.orders_history[-1])
        self.reward = self.get_reward(action, current_price)
        # print('reward++++++++')
        # print(self.reward)
        self.episode_reward += self.reward
        self.net_worth_lst.append(self.net_worth)  # for visualization
        self.reward_lst.append(self.reward)  # for visualization
        # TODO: this feel useless
        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        # info = [self.trades, self.net_worth]
        return obs, self.reward, done

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

    def get_reward(self, action, current_price):
        '''Calculates the reward for the agent.
        '''
        # ========REWARD FROM PAPER make a reference in the code if you use this=========
        # TODO: modify penalty
        if action == 1 and self.balance > self.initial_balance*0.05:
            self.number_of_purchases += 1
            self.number_of_holds = 0
            if self.number_of_purchases > 20:
                self.reward -= self.net_worth*0.1
        elif action == 0:
            self.number_of_holds += 1
            if self.number_of_holds > 20:
                self.reward -= self.net_worth*0.1
        elif action == 2 and len(self.trades) > 1 and self.trades[-1]['type'] == "sell" and self.net_worth > self.initial_balance*0.05:
            self.number_of_holds = 0
            self.number_of_purchases = 0
            profits = self.trades[-1]['profits']
            # print('PROFICTS=========')
            # print(profits)
            self.reward = profits
        return self.reward

        # ============COMPOUNDED RETURN AGAINST BUY AND HOLD==============
        # treasury_US = pd.read_csv('10-year-treasury-bond-rate-yield-chart.csv')
        # Ri_lst = []
        # Rf_lst = []
        # Rt_lst = []
        # # TODO: this can be optimized
        # if len(self.trades) > 1:
        #     for t in range(1, self.current_step):
        #         Ri = np.log(self.df.loc[t,
        #                                 'Close']) - np.log(self.df.loc[t - 1, 'Close'])
        #         Ri_lst.append(Ri)
        #         # Risk free rate
        #         # TODO:Confirm you are looking at gains
        #         start_date_row = treasury_US[treasury_US['date']
        #                                      == self.df.loc[t, 'Date']]
        #         end_date_row = treasury_US[treasury_US['date']
        #                                    == self.trades[0]['Date']]
        #         start_date_row = start_date_row.reset_index()
        #         end_date_row = end_date_row.reset_index()
        #         Rf = end_date_row.loc[0, 'value'] - \
        #             start_date_row.loc[0, 'value']
        #         Rf_lst.append(Rf)
        #         # Return for buy and hold
        #         Rt = self.df.loc[t, 'Close'] - list(self.trades)[0]['Close']
        #         Rf_lst.append(Rt)
        #     # buy
        #     # TODO: confirm these equations
        #     if action == 1:
        #         R = sum(Ri_lst) + \
        #             len(self.orders_history) * \
        #             ((1-self.trading_fee)/1+self.trading_fee)

        #     # sell
        #     elif action == 2:
        #         R = sum(Rf_lst) + \
        #             len(self.orders_history) * \
        #             np.log((1-self.trading_fee)/(1+self.trading_fee))
        #     # Return for buy and hold
        #     Rbh = sum(Rt_lst) + \
        #         np.log((1-self.trading_fee)/(1-self.trading_fee))
        #     reward = R - Rbh
        # else:
        #     reward = 0
        # return reward
        # Reward is the diff between total trading profits in percent - total eth gains in percent
        # TODO: maked sure the reward functions is doing the proper calculations
        # TODO: Maybe make this buy and hold just a daily gain instead of since the start
        # Only give reward on sell?
        # =====================CUSTOM REWARD 1==========================================
        # if self.episode_orders > 2:
        #     # buy and hold gains
        #     eth_bought_bh = (self.initial_balance - (self.initial_balance *
        #                                              self.trading_fee))/self.df.loc[self.current_step, 'Close']
        #     bh_gains = (eth_bought_bh *
        #                 self.df.loc[self.current_step, 'Close'] - (eth_bought_bh *
        #                                                            self.df.loc[self.current_step, 'Close'] * self.trading_fee)) - \
        #         eth_bought_bh*self.df.loc[0, 'Close']
        #     # trade gains
        #     # only calculated on sells
        #     if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
        #         profits = (self.trades[-2]['total'] *
        #                    self.trades[-2]['Close']) - (self.trades[-1]['Close']*self.trades[-1]['total'])
        #     elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
        #         profits = (self.trades[-1]['total'] *
        #                    self.trades[-1]['Close']) - (self.trades[-2]['Close']*self.trades[-2]['total'])

        #     self.total_trade_profits += profits
        #     reward = self.total_trade_profits - bh_gains
        #     # reward = self.total_trade_profits
        # else:
        #     # to avoid getting stuck at 0 with no orders make it hurt
        #     reward = 0
        # return reward
        # =====================CUSTOM REWARD 2 FROM: RECOMMENDING CRYPTO TRADING POINTS PAPER==========================================
        # this one needs to be coded within the buy and sell block of code in the step() function.
        # if self.episode_orders > 2 and self.episode_orders > self.prev_episode_orders:
        #     self.prev_episode_orders = self.episode_orders
        #     if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
        #         reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - \
        #             self.trades[-2]['total']*self.trades[-1]['current_price']
        #         self.trades[-1]["Reward"] = reward
        #         return reward
        #     elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
        #         reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - \
        #             self.trades[-2]['total']*self.trades[-2]['current_price']
        #         self.trades[-1]["Reward"] = reward
        #         return reward
        # else:
        #     return 0
        # # punish the bot for only holding
        # self.punish_value += self.net_worth * 0.00001
        # if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
        #     self.prev_episode_orders = self.episode_orders
        #     if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
        #         reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - \
        #             self.trades[-2]['total']*self.trades[-1]['current_price']
        #         reward -= self.punish_value
        #         self.punish_value = 0
        #         self.trades[-1]["Reward"] = reward
        #         return reward
        #     elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
        #         reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - \
        #             self.trades[-2]['total']*self.trades[-2]['current_price']
        #         reward -= self.punish_value
        #         self.punish_value = 0
        #         self.trades[-1]["Reward"] = reward
        #         return reward
        # else:
        #     return 0 - self.punish_value
