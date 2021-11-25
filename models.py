'''Methods the agent uses to trade in the environment

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Added function descriptions.
'''
import numpy as np


def Random_games(env, visualize, train_episodes=50, training_batch_size=500):
    '''The agent picks times to sell and buy the currency at random.

        - env: the gym environment the agent will learn to act in.
        - train_episodes: the number of episodes the agent will use to train.
        - training_batch_size: the ammount of steps per episode 
    '''
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        while True:
            env.render(visualize)

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)
