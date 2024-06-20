'''Agent trading strategies.

    Modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    Author: Roberto Lentini
    Email: roberto.lentini@mail.utoronto.ca
'''

import numpy as np
from utils import performance_plots, trading_chart
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras import backend as K
from collections import deque
import pandas as pd
from datetime import datetime as dt

def Random_games(env, visualize, train_episodes=50, training_batch_size=1000, comment=""):
    '''The agent picks times to sell and buy the currency at random. This is the baseline method
        to test whether the agent using our strategy is at a minimum outperforming a random buy-sell pattern.

        Args:
            env: The gym environment the agent will learn to act in.
            train_episodes (int): The number of episodes the agent will use to train.
            training_batch_size (int): Max number of steps per episode.
            comment (str): Comment for the test results.
    '''
    no_profit_episodes = 0
    average_orders = 0
    average_net_worth = 0
    avg_episode_reward_list = []
    net_worth_list = []
    for episode in range(train_episodes):
        state = env.reset()

        while True:
            env.render(visualize)

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_orders += env.episode_orders
                average_net_worth += env.net_worth
                avg_episode_reward_list.append(env.episode_reward / training_batch_size)
                net_worth_list.append(env.net_worth)
                if env.net_worth < env.initial_balance:
                    no_profit_episodes += 1
                    print(f"episode: {episode}, net_worth: {env.net_worth}, average_net_worth: {average_net_worth/(episode+1)}, orders: {env.episode_orders}")
                break
        if episode % 5 == 0:
            orders_data = pd.DataFrame.from_dict(env.trades)
            trading_chart(env, order_data=orders_data, episode=episode, price_data=env.df, filename="Random_Model", reward_annotations=False)
    with open("test_results.txt", "a+") as results:
        current_date = dt.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {"Random games"}, test episodes:{train_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/train_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, comment: {comment}\n')
    performance_plots(avg_episode_reward_list, net_worth_list, train_episodes)

    print("average_net_worth:", average_net_worth / train_episodes)

# Leads to faster performance
tf.compat.v1.disable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

class Shared_Model:
    '''An architecture where the data goes through a shared model
        prior to going through the actor and critic models.

        About the model: PPO is an actor-critic model (A2C) with a handful of changes:
        1/ Trains by using a small batch of experiences. The batch is used to
            update the policy. A new batch is then sampled and the process continues.
            This slowly changes the policy to a better version.
        2/ New formula to estimate policy gradient. Now uses the ratio between
            the new and the old policy scaled by the advantage.
        3/ New formula for estimating advantage.
    '''

    def __init__(self, input_shape, action_space, lr, optimizer, model="Dense"):
        '''Initialize the shared model with the given parameters.

        Args:
            input_shape (tuple): Shape of the input data.
            action_space (int): Size of the action space.
            lr (float): Learning rate for the optimizer.
            optimizer: Optimizer for training the model.
            model (str): Model type for the agent.
        '''
        X_input = Input(input_shape)
        self.action_space = action_space

        # Shared CNN layers
        if model == "CNN":
            X = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
            X = MaxPooling1D(pool_size=2)(X)
            X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)
            X = MaxPooling1D(pool_size=2)(X)
            X = Flatten()(X)

        # Shared LSTM layers
        elif model == "LSTM":
            X = LSTM(512, return_sequences=True)(X_input)
            X = LSTM(256)(X)

        # Shared Dense layers
        else:
            X = Flatten()(X_input)
            X = Dense(512, activation="relu")(X)

        # Critic model
        V = Dense(512, activation="relu")(X)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

        # Actor model
        A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64, activation="relu")(A)
        output = Dense(self.action_space, activation="softmax")(A)

        self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
        print(self.Actor.summary())

    def ppo_loss(self, y_true, y_pred):
        '''Calculate the PPO loss.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Loss value.
        '''
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:, 1 + self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        '''Predict the action using the actor model.

        Args:
            state (ndarray): The current state.

        Returns:
            ndarray: Predicted action.
        '''
        return self.Actor.predict(state)

    def critic_PPO2_loss(self, y_true, y_pred):
        '''Calculate the critic loss.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Loss value.
        '''
        value_loss = K.mean((y_true - y_pred) ** 2)
        return value_loss

    def critic_predict(self, state):
        '''Predict the value using the critic model.

        Args:
            state (ndarray): The current state.

        Returns:
            ndarray: Predicted value.
        '''
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

class Actor_Model:
    '''The actor is responsible for selecting actions based on the current policy. It maps states to a specific action to take.
        In neural network terms, the actor outputs a probability distribution over possible actions given the current state (in a discrete action space) 
        or directly outputs the action values (in a continuous action space).
        
        About the model: PPO is an actor-critic model (A2C) with a handful of changes:
        1/ Trains by using a small batch of experiences. The batch is used to
            update the policy. A new batch is then sampled and the process continues.
            This slowly changes the policy to a better version.
        2/ New formula to estimate policy gradient. Now uses the ratio between
            the new and the old policy scaled by the advantage.
        3/ New formula for estimating advantage.
    '''

    def __init__(self, input_shape, action_space, lr, optimizer):
        '''Initialize the actor model with the given parameters.

        Args:
            input_shape (tuple): Shape of the input data.
            action_space (int): Size of the action space.
            lr (float): Learning rate for the optimizer.
            optimizer: Optimizer for training the model.
        '''
        X_input = Input(input_shape)
        self.action_space = action_space
        X = Flatten(input_shape=input_shape)(X_input)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(64, activation="relu")(X)
        output = Dense(self.action_space, activation="softmax")(X)
        self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))

    def ppo_loss(self, y_true, y_pred):
        '''Calculate the PPO loss.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Loss value.
        '''
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:, 1 + self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.01

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        '''Predict the action using the actor model.

        Args:
            state (ndarray): The current state.

        Returns:
            ndarray: Predicted action.
        '''
        return self.Actor.predict(state)

class Critic_Model:
    '''The critic evaluates the action taken by the actor by computing a value function. 
        The critic provides feedback to the actor about how good or bad the action was, 
        which helps the actor update its policy to select better actions in the future.

        About the model: PPO is an actor-critic model (A2C) with a handful of changes:
        1/ Trains by using a small batch of experiences. The batch is used to
            update the policy. A new batch is then sampled and the process continues.
            This slowly changes the policy to a better version.
        2/ New formula to estimate policy gradient. Now uses the ratio between
            the new and the old policy scaled by the advantage.
        3/ New formula for estimating advantage.
    '''

    def __init__(self, input_shape, action_space, lr, optimizer):
        '''Initialize the critic model with the given parameters.

        Args:
            input_shape (tuple): Shape of the input data.
            action_space (int): Size of the action space.
            lr (float): Learning rate for the optimizer.
            optimizer: Optimizer for training the model.
        '''
        X_input = Input(input_shape)
        V = Flatten(input_shape=input_shape)(X_input)
        V = Dense(512, activation="relu")(V)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        '''Calculate the critic loss.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Loss value.
        '''
        value_loss = K.mean((y_true - y_pred) ** 2)
        return value_loss

    def predict(self, state):
        '''Predict the value using the critic model.

        Args:
            state (ndarray): The current state.

        Returns:
            ndarray: Predicted value.
        '''
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

def train_agent(env, agent, visualize=False, train_episodes=50, training_batch_size=500):
    '''Trains the agent using PPO.

        Args:
            env: The gym environment the agent will learn to act in.
            agent: The agent to be trained.
            visualize (bool): Flag to enable visualization.
            train_episodes (int): Number of episodes to train the agent.
            training_batch_size (int): Max number of steps per episode.
    '''
    agent.create_writer(env.initial_balance, env.normalize_value, train_episodes)
    total_average = deque(maxlen=100)
    best_average = -1000000000000000
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = agent.act(state)
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
        a_loss, c_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.episode_reward)
        average = np.average(total_average)

        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)

        print(f'episode: {episode} net worth: {env.net_worth} n_orders: {env.episode_orders} reward: {env.episode_reward}')
        if train_episodes > len(total_average):
            if best_average < env.episode_reward and env.episode_orders > 2:
                orders_data = pd.DataFrame.from_dict(env.trades)
                if len(orders_data) > 0:
                    trading_chart(env, order_data=orders_data, episode=episode, filename="train_", price_data=env.df, reward_annotations=False)
                best_average = env.episode_reward
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average), args=[episode, average, env.episode_orders, a_loss, c_loss])
            agent.save()

    agent.end_training_log()

def test_agent(env, agent, visualize=True, test_episodes=10, folder="", name="Crypto_trader", comment=""):
    '''Test agent on unseen data.

        Args:
            env: The gym environment the agent will be tested in.
            agent: The agent to be tested.
            visualize (bool): Flag to enable visualization.
            test_episodes (int): Number of episodes to test the agent.
            folder (str): Folder containing the saved model.
            name (str): Name of the saved model.
            comment (str): Comment for the test results.
    '''
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render()
            action, prediction = agent.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance:
                    no_profit_episodes += 1
                print(f"episode: {episode:<5}, net_worth: {env.net_worth:<7.2f}, average_net_worth: {average_net_worth/(episode+1):<7.2f}, orders: {env.episode_orders}")
                break
        orders_data = pd.DataFrame.from_dict(env.trades)
        if len(orders_data) > 0:
            trading_chart(env, order_data=orders_data, episode=episode, price_data=env.df)

    print(f"average {test_episodes} episodes agent net_worth: {average_net_worth/test_episodes}, orders: {average_orders/test_episodes}")
    print(f"No profit episodes: {no_profit_episodes}")
    with open("test_results.txt", "a+") as results:
        current_date = dt.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')
