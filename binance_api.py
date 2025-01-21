from binance.client import Client
import pandas as pd
import datetime

def historical_data(api_key: str, api_secret: str, symbol: str, interval: str, start_date: tuple, end_date: tuple) -> pd.DataFrame:
    """
    Récupère les données historiques d'un actif sur Binance.
    
    Paramètres:
    - api_key (str): Clé API Binance.
    - api_secret (str): Clé secrète API Binance.
    - symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT').
    - interval (str): Intervalle des bougies sous forme de string (ex: '1ms', '1s', '1m', '1h', '1d').
    - start_date (tuple): Date de début (année, mois, jour, heure, minute, seconde, milliseconde).
    - end_date (tuple): Date de fin (année, mois, jour, heure, minute, seconde, milliseconde).
    
    Retourne:
    - pd.DataFrame: Données historiques sous forme de DataFrame pandas.
    """
    interval_mapping = {
        '1s': Client.KLINE_INTERVAL_1SECOND,
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH
    }
    
    if interval not in interval_mapping:
        raise ValueError(f"Intervalle invalide: {interval}. Utilisez l'une des valeurs suivantes: {list(interval_mapping.keys())}")
    
    client = Client(api_key, api_secret)
    start_str = datetime.datetime(*start_date).strftime('%Y-%m-%d %H:%M:%S.%f')
    end_str = datetime.datetime(*end_date).strftime('%Y-%m-%d %H:%M:%S.%f')
    
    klines = client.get_historical_klines(symbol=symbol, interval=interval_mapping[interval], start_str=start_str, end_str=end_str)
    
    df = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume']
    for col in columns_to_convert:
        df[col] = df[col].astype(float)
    
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
    df = df.set_index('Open Time')
    
    return df
