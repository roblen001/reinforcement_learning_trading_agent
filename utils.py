'''Helper functions.

    modified from: https://github.com/pythonlessons/RL-Bitcoin-trading-bot
    author: Roberto Lentini
    email: roberto.lentini@mail.utoronto.ca
    date: November 24th 2021

    modifications:
        - Added function descriptions.
'''

import os
from datetime import datetime


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
