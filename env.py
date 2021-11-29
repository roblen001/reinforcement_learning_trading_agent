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
from numpy.core.arrayprint import _leading_trailing
import pandas as pd
import numpy as np
import random
from collections import deque
from gym import spaces
from utils import Write_to_file, TradingGraph
from models import Actor_Model, Critic_Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorboardX import SummaryWriter

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
        self.action_space = np.array([0, 1, 2])

        # Neural Networks part bellow
        self.lr = 0.00001
        self.epochs = 1
        self.normalize_value = 100000
        self.optimizer = Adam

        # Create Actor-Critic network model
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
   
    def create_writer(self):
        '''Tensor board writer.
        '''
        self.replay_count = 0
        self.writer = SummaryWriter(comment="Crypto_trader")

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
        self.episode_orders = 0
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
            self.episode_orders += 1


        # Sell 100% of current crypto held TODO: confirm the math for balance
        elif action == 2 and self.crypto_held > 0:
            self.crypto_sold = self.crypto_held
            self.balance += (self.crypto_sold * current_price) - \
                ((self.crypto_sold * current_price) * self.trading_fee)
            self.crypto_held -= self.crypto_sold
            self.trades.append(
                {'Date': Date, 'High': High, 'Low': Low, 'total': self.crypto_sold, 'type': "sell"})
            self.episode_orders += 1

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

    # TODO: find a new file home for this stuff it doesnt belong in the env file
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
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
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        #discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        # Compute advantages
        #advantages = discounted_r - values
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(target,'-')
        pylab.plot(advantages,'.')
        ax=pylab.gca()
        ax.grid(True)
        pylab.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1
        
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader"):
        # save keras model weights
        self.Actor.Actor.save_weights(f"{name}_Actor.h5")
        self.Critic.Critic.save_weights(f"{name}_Critic.h5")

    def load(self, name="Crypto_trader"):
        # load keras model weights
        self.Actor.Actor.load_weights(f"{name}_Actor.h5")
        self.Critic.Critic.load_weights(f"{name}_Critic.h5")

def train_agent(env, visualize=False, train_episodes = 50, training_batch_size=500):
    env.create_writer() # create TensorBoard writer
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = env.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state
            
        env.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)
        
        env.writer.add_scalar('Data/average net_worth', average, episode)
        env.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
        
        print("net worth {} {:.2f} {:.2f} {}".format(episode, env.net_worth, average, env.episode_orders))
        # taking the best performance to use for testing
        # TODO: might need to change this
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                env.save()

def test_agent(env, visualize=True, test_episodes=10):
    env.load() # load the model
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, prediction = env.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", episode, env.net_worth, env.episode_orders)
                break
            
    print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))