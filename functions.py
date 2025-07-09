# FUNCTIONS

#------ Binance access
def binance_wallet(live_trade=False):
    """
    Description: function that fetches Binance balance for the user

    Inputs: live_trade - bool, default False. If True checks for live trade ability (not needed for backtesting)
    
    Outputs objects: wallet - pd.DataFrame, existing assets and balances
                    cliente - object, binance.client.Client (EXCLUDED FROM return)
                    infos - dict, overall informations about the client (EXCLUDED FROM return)
    """
    # Necessary packages
    import pandas as pd
    from binance.client import Client  # Binance
    from keys import api_secret_offline, api_key_offline  # These are your infos, read the README.md if you're lost here

    def request_wallet():
            # Requests user wallet infos
            print('Fetching wallet balance...')
            wallet = pd.DataFrame(infos['balances'])

            # Gets the 'numerical' informations about the balances
            nums = ['free', 'locked']

            # Transform objs in float
            wallet[nums] = wallet[nums].astype(float)

            # Filter the assets with balance
            mask = wallet[nums][wallet[nums] > 0].dropna(how='all').index
            print('Cleaning wallet from non-positive cryptos...')
            wallet = wallet.iloc[mask]  # keep only assets with positive balance

            # If needed, excludes some cryptos (asset blacklisting)
            black_list = ['NFT', 'SHIB', 'BTTC']
            mask = wallet[wallet['asset'].isin(black_list)].index  # blacklist index
            wallet.drop(mask, axis=0, inplace=True)  # dropping blacklist
            print('Done.')

            print(f'\n--> Please note this account type: {infos['accountType']} <--')

            wallet.reset_index(drop=True, inplace=True)

            return wallet
    
    # Get overall info from Binance, using offline (not live) keys
    cliente = Client(api_key_offline, api_secret_offline)

    # Checks for systems online
    if cliente.get_system_status()['msg'] != 'normal':
        print('\n\n!!!! **** WARNING **** !!!!\n')
        print('!!!! BINANCE OFFLINE !!!!\n')
        print('Unable to fetch data\n\n')

    else:
        print('\nBinance on-line. Requesting data.')
        # Fetch user data
        infos = cliente.get_account()

        if live_trade == True:
            # Check if the user is able to live trade (not mandatory for offline)
            if infos['canTrade'] == False:
                print('\nWARNING! User unable to trade, please check status with Binance!')
                print('Aborting.')
            else:
                wallet = request_wallet()

        else:
            wallet = request_wallet()
    
    return wallet


#------ Historical data
def historical_data(ticker='BTCUSDT', days=30, interval='15m'):  # the ticker in Binance works in pairs - here you'll want to know how much is BTC worth in USDT, for example
    """
    Description: gets the trading pair historical data from Binance.

    Inputs: ticker - str, default 'BTCUSDT', the pair you want to trade;
            days - int, default 30, gets historical data from this many days ago;
            interval - str, default '15m', the 'slicing' of the timeframe, more info at https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams#klinecandlestick-streams-for-utc
    
    Outputs objects: hist - pd.DataFrame, OHLC historical data
    """
    import datetime
    import requests
    import json
    import time
    import pandas as pd

    # Defines the data timespan
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=days)
    
    # Converts time to Unix (because Binance) in miliseconds
    end_timestamp = int(end_time.timestamp()*1000)
    start_timestamp = int(start_time.timestamp()*1000)

    # Binance endpoint
    endpoint = 'https://api.binance.com/api/v3/klines'

    # Timewindow estabilished, requests historical data.
    # Request parameters.
    limit = 1000
    params = {'symbol': ticker, 'interval': interval,
          'endTime': end_timestamp, 'limit': limit,
          'startTime': start_timestamp}
    print('Requesting informations from Binance.')

    # Make the request and saves it in a list. 'Dados' means 'data' in portuguese.
    dados = []
    while True:
        response = requests.get(endpoint, params=params)
        klines = json.loads(response.text)
        dados += klines
        if len(klines) < limit:
            break
        params['startTime'] = int(klines[-1][0])+1
        time.sleep(0.1)
    print('Request successful. Splitting data...')

    # Pick specific data from fetched data
    # About kline[n] pos: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints
    loose_data = []
    for kline in dados:
        loose_data = [[float(kline[1]), float(kline[2]), float(kline[3]), float(kline[4]), float(kline[5])] for kline in dados]

    # Creates the DataFrame
    timestamps = [datetime.datetime.fromtimestamp(int(kline[0])/1000) for kline in dados]
    hist = pd.DataFrame(loose_data, columns=['open', 'high', 'low', 'close', 'volume'], index=timestamps)
    hist = pd.concat([hist], keys=[ticker], names=['asset', 'time'])

    print('All done.')

    return hist


#------ Check for trading pairs CSV file
def check_pairs():
    """
    Description: checks if there's a trading pairs record, makes up the trading pairs from Binance toAsset and fromAsset data if none is found
    Inputs: none
    Outputs: none (generates a csv file though)
    """
    import datetime
    import requests
    import json
    import os
    import pandas as pd

    # Stores current month and year, used to validate files
    today = datetime.datetime.now().strftime('%Y%m')
    
    # Resources and backup path
    resources_dir = './resources/'
    backup_dir = 'older_versions/'

    #------ List with the 20 top market cap currencies
    def top20():
        """
        Description: get the 20 biggest market cap cryptos from the web. Needs tweaks for each source.

        input: none

        outpu: list, cryptocurrencies symbols
        """
        import requests
        from bs4 import BeautifulSoup

        cmc = 'https://crypto.com/price'
        
        print('Getting list of the 20 cryptos with the most market cap.')

        try:
            response = requests.get(cmc)
            soup = BeautifulSoup(response.text, "html.parser")

            site = soup.find_all("span", {"class": "chakra-text"})

            cryptos = []
            for cur in site:
                cryptos.append(cur.get_text())
            
            print('Done.')
        
        except:
            print(f"Error fetching biggest market cap cryptos from {cmc}")
        
        return cryptos[0:20]


    #------ Creates the buy/sell pairs from Binance endpoint
    def create_pairs_file():
        # Sets Binance endpoint and its parameter (crypto common to all trades, in this case)
        pairs_endpoint = 'https://api.binance.com/sapi/v1/convert/exchangeInfo'
        params = {'toAsset': 'USDT'}

        # Fetches the Top 20 market cap cryptos from the web to make our asset basket
        crypto_list = top20()

        # Makes the request
        print('Retrieving information about pairing trades from Binance.')
        response = requests.get(pairs_endpoint, params=params)
        
        # Changes the response into a DataFrame
        df = pd.DataFrame(json.loads(response.text))

        # Filters Binance cryptos data maintaining only the top 20 at most
        print('Filtering all assets not tradable from Binance.')
        mask = df['fromAsset'].isin(crypto_list)
        df = df.loc[mask].reset_index(drop=True)

        # Creates the buy/sell pairs
        print('Registering tradable pairs.')
        sell_pairs = df['fromAsset'] + df['toAsset']
        buy_pairs = df['toAsset'] + df['fromAsset']

        # Creates a new temp DataFrame with just the pairs
        temp = pd.concat([buy_pairs, sell_pairs], axis=1)
        temp.columns = ['Buy', 'Sell']

        # Saves to file
        print('Creating new file...')
        temp.to_csv(f'{resources_dir}pairs_{today}.csv', index=False)
        print(f"Trading pairs file created: '{resources_dir}pairs_{today}.csv'")
        print('All done.')        

    # Checks for path
    if not os.path.exists(resources_dir):
        print(f"Directory '{resources_dir}' does not exist. Let's make it, shall we?")
        os.mkdir(resources_dir)
        print(f"Done, directory '{resources_dir}' created successfully.")
    
    else:
        print('The resources directory exists, checking for trade pairs file.')

    # Check for any pair file. If it exists and is newer than a month, loads it.
    arqs = os.listdir(resources_dir)

    # Keeps only generated files, disregarding folders or other misc files
    for i in arqs:
        if 'pairs' not in str(i):
            arqs.remove(i)

    if len(arqs) == 0:
        print('Trading file not found in folder. Creating...')
        create_pairs_file()

    elif len(arqs) == 1:
        arqs = arqs[0]
        print(f"Trading pairs file '{arqs}' found, checking version.")
        file_found = arqs.split('.csv')[0]
        file_found = file_found.split('_')[1]

        if int(today) > int(file_found):
            import shutil                        
            print(f"Time to update the files! Moving current file to './{backup_dir}'.")
            shutil.move(resources_dir+arqs, resources_dir+backup_dir+arqs)
            print(f'Updating trading pairs file.')
            create_pairs_file()

        else:
            print('Trading pairs file is up to date.')

    elif len(arqs) > 1:
        print(f"WARNING: Multiple files found! Moving all trading pairs files to './{backup_dir}'.")
        for i in arqs:
            import shutil
            shutil.move(resources_dir+i, resources_dir+backup_dir+i)
        print(f'Creating valid trading pairs file.')
        create_pairs_file()
    
    else:
        print("Some kind of witchcraft error happened. This message isn't supposed to show up!")


#------ Loads the trading pairs CSV file
def get_pairs():
    """
    Description: loads the latest trading pairs file using pandas.read_csv()

    Input: none
    Output: pandas.DataFrame
    """
    import os
    import pandas as pd

    resources_dir = './resources/'

    # Check for the pairs file.
    arqs = os.listdir(resources_dir)

    # Keeps only generated files, disregarding folders or other misc files
    for i in arqs:
        if 'pairs' not in str(i):
            arqs.remove(i)

    if len(arqs) == 0:
        print('ERROR: Trading file not found in folder. Please run check_pairs() first.')

    elif len(arqs) == 1:
        arqs = arqs[0]
        pairs = pd.read_csv(f'{resources_dir}{arqs}')

    elif len(arqs) > 1:
        print(f"ERROR: Multiple files found! Please run check_pairs() first.")

    return pairs


#------ ATR indicator estimation
def atr_calc(df, length=20):
    """
    Description: function to compute Average True Range estimations.
    Important Notice: df MUST have 'high', 'low', 'close' values features.

    Input: df, OHLC pandas DataFrame
    Output: pandas Series, with standardized ATR
    """
    import pandas_ta

    atr = pandas_ta.atr(high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        length=length)
    
    return atr.sub(atr.mean()).div(atr.std())


#------ MACD estimation
def macd_calc(df, length=30):
    """
    Description: custom function to estimate MACD indicator
    Inputs: df, OHLC pandas DataFrame
    Outputs: indicator estimated
    """
    import pandas_ta

    macd = pandas_ta.macd(close=df['close'], length=length)

    return macd.sub(macd.mean()).div(macd.std())


#------ Return estimation
def estimate_returns(df):
    """
    Definition: estimate cumulative monthly returns (up to 6m)
    Input: DataFrame containing OHLC 'close' infos, fortnightly
    Outputs: updated DataFrame
    """
    outlier = 0.005 

    lags = [2, 4, 6, 8, 10, 12]

    for time in lags:
        df[f'return_{int(time/2)}m'] = df['close'].pct_change(time).pipe(lambda x: x.clip(lower=x.quantile(outlier), upper=x.quantile(1-outlier))).add(1).pow(1/time).sub(1)

    return df





