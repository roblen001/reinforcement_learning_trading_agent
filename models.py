'''Methods the agent uses to trade in the environment

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Added function descriptions.
        - Added performance visualization.
'''
import numpy as np
from utils import performance_plots
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import backend as K

# ================== RANDOM ORDERS =================


def Random_games(env, visualize, train_episodes=50, training_batch_size=500):
    '''The agent picks times to sell and buy the currency at random.

        - env: the gym environment the agent will learn to act in.
        - train_episodes: the number of episodes the agent will use to train.
        - training_batch_size: the ammount of steps per episode
    '''
    average_net_worth = 0
    average_reward = 0
    avg_episode_reward_list = []
    net_worth_list = []
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        while True:
            env.render(visualize)

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                # for graphing
                avg_episode_reward_list.append(
                    env.episode_reward/training_batch_size)
                # self.net_worth is the final networth after each episode
                net_worth_list.append(env.net_worth)
                break

    performance_plots(avg_episode_reward_list,
                      net_worth_list, train_episodes)

    print("average_net_worth:", average_net_worth/train_episodes)


# tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
# usually using this for fastest performance
tf.compat.v1.disable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

# ================== RANDOM ORDERS =================


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
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
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:,
                                                                      1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING,
                    max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)

        V = Flatten(input_shape=input_shape)(X_input)
        V = Dense(512, activation="relu")(V)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs=value)
        self.Critic.compile(loss=self.critic_PPO2_loss,
                            optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2)  # standard PPO loss
        return value_loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
