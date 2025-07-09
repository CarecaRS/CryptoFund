# CryptoFund

This is an approach to a cryptocurrencies fund (a.k.a 'The Fund') using Binance as the exchange and an user-defined number of assets to invest in, based on the top 20 crypto pairs traded over Binance, using exclusively Python (version 3.13.3 at the time of this writing). The 'top 20' classification is based on the biggest market cap assets, get from anywhere over the internet (current script fetches from crypto.com). The Fund will have two different strategies: (1) passive, trade on 1st of each month, keep in basket the 10 cryptos with the most market cap; (2) active, trade every week based on predicted returns for the following week.

The main file is coded using Jupyter Notebook for the ease of use by the newcomers. If you decide to put it into production (trading with actual money) I advise you to transform this file to a simple Python script (`foo.py`), run it from your terminal (local machine or somewhere in the cloud - recommended) and keep it running in the background. This algorithm can trade the currencies paired only with USDT (available here in https://www.binance.us/price). 

As stated, the CryptoFund algo uses Binance to trade cryptos. But it is easily changed to trade stocks, derivatives or whichever other asset class in any exchange that the user needs, you just have to tweak some parts of the code. Use this as a template and have fun with it.


## Required Python packages
Install by `pip3 install [package/package list]` or whatever way you find best. Some of them are already built-in, but every (potentially) imported package throughout the script are the following:

- `pandas`
- `pandas_ta`
- `numpy`
- `python-binance`
- `time`
- `datetime`
- `requests`
- `json`
- `os`
- `beautifulsoup4`

# DISCLAIMEMR
THIS BOT IS NOT AN INVESTMENT/TRADING ADVICE!

The Fund can perform trades with real money, yes, but it is used just for educational purposes and should not be used to try to make any profit or be part of any portfolio whatsoever.

If you should use my system (or any variation of it) please be aware that you're on your risk and I'm not responsible for your profits or losses.

Even if you happened to build a strategy that got favorable results with historical data (e.g. returned profit) it absolutely does not guarantee that you'll get the same results with real-time data.

Don't come whining to me if you lost your rent money.