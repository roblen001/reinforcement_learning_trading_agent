'''Helper functions.

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Added function descriptions.
        - Adding function for performance evaluation of agent.
        - removing the render graph function, we only want to display a graph at the end.
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

# ================ COMPLETLY CHANGE THIS ================================
# TODO: add labels with reward values in the graph


class TradingGraph:
    '''A crypto trading visualization using matplotlib made to render custom prices which come in following way:
        Date, Open, High, Low, Close, Volume, net_worth, trades
        call render every step.
    '''

    def __init__(self, Render_range):
        self.Volume = deque(maxlen=Render_range)
        self.net_worth = deque(maxlen=Render_range)
        self.render_data = deque(maxlen=Render_range)
        self.Render_range = Render_range

        plt.style.use('ggplot')
        # close all plots if they're are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16, 8))

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid(
            (6, 1), (5, 0), rowspan=1, colspan=1, sharex=self.ax1)

        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
        # self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')

        # Add paddings to make graph easier to view
        # plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

    # Render the environment to the screen
    def render(self, Date, Open, High, Low, Close, Volume, net_worth, trades):
        # append volume and net_worth to deque list
        self.Volume.append(Volume)
        self.net_worth.append(net_worth)

        # before appending to deque list, need to convert Date to special format
        Date = mpl_dates.date2num([pd.to_datetime(Date)])[0]
        self.render_data.append([Date, Open, High, Low, Close])

        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8/24,
                         colorup='green', colordown='red', alpha=0.8)

        # Put all dates to one list and fill ax2 sublot with volume
        Date_Render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(Date_Render_range, self.Volume, 0)

        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(Date_Render_range, self.net_worth, color="blue")

        # beautify the x-labels (Our Date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            if trade_date in Date_Render_range:
                if trade['type'] == 'buy':
                    high_low = trade['Low']-10
                    self.ax1.scatter(trade_date, high_low, c='green',
                                     label='green', s=120, edgecolors='none', marker="^")
                else:
                    high_low = trade['High']+10
                    self.ax1.scatter(trade_date, high_low, c='red',
                                     label='red', s=120, edgecolors='none', marker="v")

        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        # plt.show(block=False)
        # Necessary to view frames before they are unrendered
        # plt.pause(0.001)

        """Display image with OpenCV - no interruption"""
        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(),
                            dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with OpenCV or any operation you like
        cv2.imshow("Bitcoin trading bot", image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
