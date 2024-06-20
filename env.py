'''This is a custom cryptocurrency trading environment created with
    OpenAI Gym. Handles the agent creation and environment creation.
    This includes the reward function, action space, and state space.

    Modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    Author: Roberto Lentini
    Email: roberto.lentini@mail.utoronto.ca
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
        '''Initialize the agent with the given parameters.

        Args:
            lookback_window_size (int): Number of steps to look back for the state.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of epochs for training.
            optimizer: Optimizer for training the model.
            batch_size (int): Batch size for training.
            model (str): Model type for the agent.
        '''
        self.lookback_window_size = lookback_window_size
        self.model = model

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        # Folder to save models
        self.log_name = dt.now().strftime("%Y_%m_%d_%H_%M") + "_Crypto_trader"

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (lookback_window_size, 20)

        # Neural Networks part below
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create shared Actor-Critic network model
        self.Actor = self.Critic = Shared_Model(
            input_shape=self.state_size, action_space=self.action_space.shape[0], lr=self.lr, optimizer=self.optimizer, model=self.model)
        
    def create_writer(self, initial_balance, normalize_value, train_episodes):
        '''Create a Tensorboard writer.

        Args:
            initial_balance (float): Initial balance for trading.
            normalize_value (float): Normalization value for the data.
            train_episodes (int): Number of training episodes.
        '''
        self.replay_count = 0
        self.writer = SummaryWriter('runs/' + self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.start_training_log(initial_balance, normalize_value, train_episodes)

    def start_training_log(self, initial_balance, normalize_value, train_episodes):
        '''Log the start of training parameters to a file.

        Args:
            initial_balance (float): Initial balance for trading.
            normalize_value (float): Normalization value for the data.
            train_episodes (int): Number of training episodes.
        '''
        with open(self.log_name + "/Parameters.txt", "w") as params:
            current_date = dt.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"Training start: {current_date}\n")
            params.write(f"Initial balance: {initial_balance}\n")
            params.write(f"Training episodes: {train_episodes}\n")
            params.write(f"Lookback window size: {self.lookback_window_size}\n")
            params.write(f"Learning rate: {self.lr}\n")
            params.write(f"Epochs: {self.epochs}\n")
            params.write(f"Batch size: {self.batch_size}\n")
            params.write(f"Normalize value: {normalize_value}\n")
            params.write(f"Model: {self.model}\n")

    def end_training_log(self):
        '''Log the end of training to a file.'''
        with open(self.log_name + "/Parameters.txt", "a+") as params:
            current_date = dt.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"Training end: {current_date}\n")

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        '''Calculate Generalized Advantage Estimation (GAE).

        Args:
            rewards (list): List of rewards.
            dones (list): List of done flags.
            values (list): List of values from the Critic network.
            next_values (list): List of next values from the Critic network.
            gamma (float): Discount factor for rewards.
            lamda (float): Smoothing parameter for GAE.
            normalize (bool): Flag to normalize the GAE values.

        Returns:
            tuple: Tuple containing the advantages and the target values.
        '''
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        '''Replay and train the agent.

        Args:
            states (list): List of states.
            actions (list): List of actions.
            rewards (list): List of rewards.
            predictions (list): List of predictions from the Actor network.
            dones (list): List of done flags.
            next_states (list): List of next states.

        Returns:
            tuple: Tuple containing the actor loss and critic loss.
        '''
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)

        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        y_true = np.hstack([advantages, predictions, actions])

        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state):
        '''Predict the next action to take using the model.

        Args:
            state (ndarray): The current state.

        Returns:
            tuple: The chosen action and the prediction probabilities.
        '''
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader", score="", args=[]):
        '''Save the model weights.

        Args:
            name (str): Name of the model.
            score (str): Score to append to the model name.
            args (list): List of additional arguments to log.
        '''
        self.Actor.Actor.save_weights(f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(f"{self.log_name}/{score}_{name}_Critic.h5")

        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = dt.now().strftime('%Y-%m-%d %H:%M:%S')
                log.write(f"{current_time}, {args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}\n")

    def load(self, folder, name):
        '''Load the model weights.

        Args:
            folder (str): Folder containing the model weights.
            name (str): Name of the model.
        '''
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(os.path.join(folder, f"{name}_Critic.h5"))

class EthereumEnv:
    """Custom Ethereum Environment that follows gym interface"""

    def __init__(self, df, initial_balance=1000, lookback_window_size=50, trading_fee=0.07, debug_mode=False, normalize_value=40000):
        '''Initialize the environment with the given parameters.

        Args:
            df (DataFrame): DataFrame containing the historical data.
            initial_balance (int): Initial balance for trading.
            lookback_window_size (int): Number of steps to look back for the state.
            trading_fee (float): Trading fee percentage.
            debug_mode (bool): Flag to enable debug mode.
            normalize_value (int): Normalization value for the data.
        '''
        super(EthereumEnv, self).__init__()
        self.df = df
        self.trading_fee = trading_fee
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.debug_mode = debug_mode
        
        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Blockchain analysis data
        self.blockchain_data = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value

    def reset(self, env_steps_size=0):
        '''Reset the environment to an initial state.

        Args:
            env_steps_size (int): Step size for training the data. An alternative to random initial offset.

        Returns:
            deque: Initial state of the environment.
        '''
        self.total_trade_profits = 0
        self.prev_episode_orders = 0  # Track previous episode orders count
        self.punish_value = 0
        self.net_worth_lst = deque()  # For visualization
        self.trades = deque()  # For visualization
        self.episode_reward = 0
        self.reward_lst = deque()  # For visualization
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0
        self.number_of_purchases = 0
        self.number_of_holds = 0
        self.number_of_sales = 0
        if env_steps_size > 0:  # Used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # Used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            if 'level_0' in self.df.columns:
                self.df = self.df.drop(['level_0'], axis=1).reset_index()
            else:
                self.df = self.df.reset_index()
            current_step = self.current_step - i

            self.orders_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

            self.blockchain_data.append([self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'receive_count'],
                                        self.df.loc[current_step, 'sent_count'],
                                        self.df.loc[current_step, 'avg_fee'],
                                        self.df.loc[current_step, 'blocksize'],
                                        self.df.loc[current_step, 'btchashrate'],
                                        self.df.loc[current_step, 'OIL'],
                                        self.df.loc[current_step, 'ecr20_transfers'],
                                        self.df.loc[current_step, 'GOLD'],
                                        self.df.loc[current_step, 'searches'],
                                        self.df.loc[current_step, 'hashrate'],
                                        self.df.loc[current_step, 'marketcap'],
                                        self.df.loc[current_step, 'difficulty'],
                                        self.df.loc[current_step, 's&p500'],
                                        self.df.loc[current_step, 'transactionfee'],
                                        self.df.loc[current_step, 'transactions'],
                                        self.df.loc[current_step, 'tweet_count'],
                                        self.df.loc[current_step, 'unique_adresses'],
                                        self.df.loc[current_step, 'VIX'],
                                        self.df.loc[current_step, 'UVYX']
                                         ])

        state = self.blockchain_data
        return state  # Reward, done, info can't be included

    def _next_observation(self):
        '''Get the data points for the given current_step state.

        Returns:
            deque: Observation for the current step.
        '''
        self.blockchain_data.append([self.df.loc[self.current_step, 'Close'],
                                     self.df.loc[self.current_step, 'receive_count'],
                                     self.df.loc[self.current_step, 'sent_count'],
                                     self.df.loc[self.current_step, 'avg_fee'],
                                     self.df.loc[self.current_step, 'blocksize'],
                                     self.df.loc[self.current_step, 'btchashrate'],
                                     self.df.loc[self.current_step, 'OIL'],
                                     self.df.loc[self.current_step, 'ecr20_transfers'],
                                     self.df.loc[self.current_step, 'GOLD'],
                                     self.df.loc[self.current_step, 'searches'],
                                     self.df.loc[self.current_step, 'hashrate'],
                                     self.df.loc[self.current_step, 'marketcap'],
                                     self.df.loc[self.current_step, 'difficulty'],
                                     self.df.loc[self.current_step, 's&p500'],
                                     self.df.loc[self.current_step, 'transactionfee'],
                                     self.df.loc[self.current_step, 'transactions'],
                                     self.df.loc[self.current_step, 'tweet_count'],
                                     self.df.loc[self.current_step, 'unique_adresses'],
                                     self.df.loc[self.current_step, 'VIX'],
                                     self.df.loc[self.current_step, 'UVYX']
                                     ])

        obs = self.blockchain_data

        return obs

    def step(self, action):
        '''Execute a step in the environment.

        Args:
            action (int): Action to take.

        Returns:
            tuple: Observation, reward, and done flag for the current step.
        '''
        self.reward = 0
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        current_price = self.df.loc[self.current_step, 'Close']
        Date = self.df.loc[self.current_step, 'Date']  # For visualization
        if action == 0:  # Hold
            pass
        elif action == 1 and self.balance > self.initial_balance * 0.05:  # Buy with 100% of current balance
            self.crypto_bought = (self.balance - (self.trading_fee * self.balance)) / current_price
            self.balance = 0
            self.crypto_held += self.crypto_bought
            self.episode_orders += 1
            self.trades.append({'Date': Date, 'Close': current_price,
                                'total': self.crypto_bought, 'type': 'buy',
                                'Net_worth': 0, 'Reward': None, 'current_price': current_price})

        elif action == 2 and self.crypto_held * current_price > self.initial_balance * 0.05:  # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += (self.crypto_sold * current_price) - ((self.crypto_sold * current_price) * self.trading_fee)
            self.crypto_held = 0
            self.episode_orders += 1
            profits = self.balance - self.trades[-1]['total'] * self.trades[-1]['Close']
            self.trades.append({'Date': Date, 'Close': current_price,
                                'total': self.crypto_sold, 'type': 'sell',
                                'Net_worth': 0, 'Reward': None, 'current_price': current_price, 'profits': profits})

        self.net_worth = self.balance + (self.crypto_held * current_price)
        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        if self.debug_mode:
            Write_to_file(Date, self.orders_history[-1])
        self.reward = self.get_reward(action, current_price)
        self.episode_reward += self.reward
        self.net_worth_lst.append(self.net_worth)  # For visualization
        self.reward_lst.append(self.reward)  # For visualization

        done = self.net_worth <= self.initial_balance / 2

        obs = self._next_observation()

        return obs, self.reward, done

    def render(self, visualize=False):
        '''Render plot and output while agent is running.

        Args:
            visualize (bool): If True, then a graph visualization will appear.
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
        '''Calculate the reward for the agent.

        Args:
            action (int): Action taken by the agent.
            current_price (float): Current price of the asset.

        Returns:
            float: Reward for the current action.
        '''
        if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
            self.prev_episode_orders = self.episode_orders
            if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
                reward = self.trades[-2]['total'] * self.trades[-2]['current_price'] - self.trades[-2]['total'] * self.trades[-1]['current_price']
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
                reward = self.trades[-1]['total'] * self.trades[-1]['current_price'] - self.trades[-2]['total'] * self.trades[-2]['current_price']
                self.trades[-1]["Reward"] = reward
                return reward
        else:
            return 0
