# CryptoFund

This is an approach to a cryptocurrencies fund using Binance as the exchange and an user-defined number of assets to invest in, based on the top 20 crypto pairs traded over Binance, using exclusively Python (version 3.13.3 at the time of this writing). The 'top 20' classification is based on the biggest market cap assets, get from [CoinMarketCap.com].

The main file is coded using Jupyter Notebook for the ease of use by the newcomers. If you decide to put it into production (trading with actual money) I advise you to transform this file to a simple Python script (foo.py) and run it from your terminal (local machine or somewhere in the cloud - recommended). This algorithm can trade the currencies paired only with USDT (available here [https://www.binance.us/price]). 

As stated, the CryptoFund algo uses Binance to trade cryptos. But it is easily changed to trade stocks or whichever other asset class in any exchange that the user needs, you just have to tweak some parts of the code. Use this as a template.

My main goal is to re-balance the assets basket every week, maybe every month or even more for longer positions.

## Required Python packages
Install by `pip3 install [package/package list]`. Some of them are already built-in, but every (potentially) imported package throughout the script are the following:

- pandas
- numpy
- python-binance
- time
- datetime
- requests
- json
- os
- beautifulsoup4