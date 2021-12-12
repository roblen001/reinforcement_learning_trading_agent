'''Helper functions.

    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

'''

import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np
import matplotlib.dates as mdates


def Write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))):
    '''Saves order history to a txt file in a log folder.
        Used for debugging.

        - net_wroth: int of the net worth.
        - file_name: str of the name where the file will be saved
    '''
    # information in file: balance, net worth, crypto bought, crypto sold
    for i in net_worth:
        Date += " {}".format(i)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
    file.write(Date+"\n")
    file.close()


def performance_plots(avg_reward, net_worth, n_episodes):
    '''Plots to visualize performance of trading agent.
        1/ Average reward per episode
        2/ Networth per episode

        - avg_reward: lst of average reward per episode.
        - net_worth: lst of final networth per episode.
        - n_episode: int of number of episodes executed.
    '''
    fig, axs = plt.subplots(2)
    fig.suptitle('Agent Performance')

    axs[0].plot(list(range(0, n_episodes)), avg_reward)
    axs[1].plot(list(range(0, n_episodes)), net_worth)
    axs[0].set(ylabel='avg reward')
    axs[1].set(ylabel='net worth')
    plt.show()


def trading_chart(env, order_data, episode, price_data, filename="", reward_annotations=False):
    '''Visualize the the trades my by the agent.

        - order_data: dataframe with the following columns:
            Date, Open, High, Low, Close, Volume,
            Type, Net_worth, Reward. Data of only time periods where
            orders were placed.
        - episode: int of current episode when graph is being plotted.
        - price_data: dataframe of data at every step of time frame.
    '''
    # TODO: Add a way to make sure the files dont save over each other
    #  we want the plots for each model
    # cleaning data to plot
    # order_data = order_data.drop(
    #     ['High', 'Low', 'Close', 'Net_worth'], axis=1)
    order_data = order_data.drop(
        ['Close', 'Net_worth'], axis=1)
    order_data['Date'] = pd.to_datetime(
        order_data['Date'])
    price_data['Date'] = pd.to_datetime(
        price_data['Date'])
    df = pd.merge(price_data, order_data, how='left', on='Date', sort=False)
    df = df[env.start_step:env.end_step]
    start_date = df.head(1)['Date']
    end_date = df.tail(1)['Date']
    # adding the networth at everytime step
    df['Net_worth'] = env.net_worth_lst
    # df['Reward'] = env.reward_lst
    # plotting figure
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))
    ax[0].plot(df['Date'], df['Close'], label='Ethereum Price')
    ax[1].plot(df['Date'], df['Close'], label='Ethereum Price')
    ax[1].plot(df['Date'], df['Net_worth'], label='Net worth')
    ax[0].text(0.3, 0.9, 'networth: ' + str(round(env.net_worth)) + ' reward: ' + str(round(env.episode_reward)), ha='center',
               va='center', transform=ax[0].transAxes,  bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    ax[0].text(3, 4, 'Random Noise', style='italic', fontsize=12,
               bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
    buys = df.query('type == "buy"')
    sells = df.query('type == "sell"')
    ax[0].scatter(buys['Date'], buys['Close'],
                  label='Buy', marker='^', color='green', alpha=1)
    ax[0].scatter(sells['Date'], sells['Close'],
                  label='Sell', marker='v', color='red', alpha=1)
    ax[0].set_title('Ethereum Price History with buy and sell signals',
                    fontsize=10, backgroundcolor='white', color='white')
    ax[0].set_ylabel('Close Price', fontsize=18)
    ax[1].set_ylabel('Close Price', fontsize=18)
    ax[0].set_title("Trades Plot")
    ax[1].set_title('Networth and Price Plot')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    # Adding labels for action rewards
    # TODO: fix annotations
    if reward_annotations == True:
        if 'level_0' in sells.columns:
            sells = sells.drop(
                ['level_0'], axis=1).reset_index()
        else:
            sells = sells.reset_index()
        for i in range(len(sells)):
            ax[0].annotate('{0:.2f}'.format(int(sells['Reward'][i])), xy=(int(mdates.date2num(sells['Date'][i])), int(sells['Close'][i])),
                           xytext=(int(mdates.date2num(sells['Date'][i])-8),
                                   int(sells['Close'][i] + 100)),
                           bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
        # if 'level_0' in buys.columns:
        #     buys = buys.drop(
        #         ['level_0'], axis=1).reset_index()
        # else:
        #     buys = buys.reset_index()
        # for i in range(len(buys)):
        #     ax[0].annotate('{0:.2f}'.format(int(buys['Reward'][i])), xy=(int(mdates.date2num(buys['Date'][i])), int(buys['Close'][i])),
        #                    xytext=(int(mdates.date2num(buys['Date'][i])-8),
        #                            int(buys['Close'][i] + 100)),
        #                    bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename + 'trades_plot_episode_' + str(episode) + '.png')
