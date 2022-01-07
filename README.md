reinforcement_learning_trading_agent


Deep Reinforcement Learning Autonomous Ethereum Trading Agent
By Roberto Lentini

A challenge with trading any kind of asset is the traders ability to time the market and make the correct decision to buy, sell, or hold an asset. Making the correct decision will help a trader yield a greater profit. Many different strategies are often used to try and time the market correctly to maximize profit. An issue with these strategies for an amateur trader is the emotion involved while making the decision which can cause them to panic sell or buy the excitement when it might not be an optimal time. A solution to this issue would be to create an autonomous trading agent which can buy and sell for the trader. In my project I will be using deep reinforcement learning to create an autonomous cryptocurrency trading agent to trade Ethereum and tie it to my pre-existing trading dashboard.

This project will combine my interest in web development as well as my previous courses in machine learning to create a usable end product. I currently have a dashborad and flask-api created and tied to a test crypto account the code can be found here: https://github.com/roblen001/Crypto-Bot. The flask-api for the project is currently running on a server and scraping a crypto news aggregator 24/7 as well as logging crypto prices and trying different basic trading strategies.

For this reinforcement learning problem the agent will be the entity making the decisions to buy and sell Ethereum. The environment will be the Ethereum currency historical data. The actions will be to buy, sell or hold the asset. The reward will be calculated when the agent sells the currency by subtracting the assets buy price to the assets sell price. 

The projects inspiration comes from the following paper: https://www.mdpi.com/2076-3417/10/4/1506 

