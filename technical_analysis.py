import pandas as pd

def cmf(high, low, close, volume, window=20):
    """
    Calculer le Chaikin Money Flow (CMF) à partir de séries temporelles.

    Paramètres :
        high (pd.Series) : Série temporelle des prix les plus hauts.
        low (pd.Series) : Série temporelle des prix les plus bas.
        close (pd.Series) : Série temporelle des prix de clôture.
        volume (pd.Series) : Série temporelle des volumes échangés.
        window (int) : La période de calcul pour le CMF (par défaut : 20).

    Retourne :
        pd.Series : Une série contenant les valeurs du CMF.
    """
    # Vérifier que toutes les séries ont la même longueur
    if not (len(high) == len(low) == len(close) == len(volume)):
        raise ValueError("Toutes les séries doivent avoir la même longueur.")
    
    # Calculer le Money Flow Multiplier (MFM)
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = mfm.fillna(0)  # Gérer les divisions par zéro ou les NaN

    # Calculer le Money Flow Volume (MFV)
    mfv = mfm * volume

    # Calculer les sommes glissantes
    sum_mfv = mfv.rolling(window=window).sum()
    sum_volume = volume.rolling(window=window).sum()

    # Calculer le CMF
    cmf = sum_mfv / sum_volume
    return cmf.fillna(0)  # Remplacer les NaN éventuels par zéro

def rsi(close, window=14):
    """
    Calculer l'Indice de Force Relative (RSI) à partir d'une série temporelle.

    Paramètres :
        close (pd.Series) : Série temporelle des prix de clôture.
        window (int) : La période de calcul pour le RSI (par défaut : 14).

    Retourne :
        pd.Series : Une série contenant les valeurs du RSI.
    """
    # Vérifier que la série est bien définie
    if close.isnull().all():
        raise ValueError("La série des prix de clôture ne peut pas être vide ou contenir uniquement des NaN.")

    # Calculer les variations des prix de clôture
    delta = close.diff()

    # Séparer les variations positives et négatives
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculer les moyennes mobiles exponentielles des gains et pertes
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculer la force relative (RS)
    rs = avg_gain / avg_loss

    # Calculer le RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(0)  # Remplacer les NaN éventuels par zéro

def obv(close, volume):
    """
    Calculer l'On-Balance Volume (OBV) à partir de séries temporelles.

    Paramètres :
        close (pd.Series) : Série temporelle des prix de clôture.
        volume (pd.Series) : Série temporelle des volumes échangés.

    Retourne :
        pd.Series : Une série contenant les valeurs de l'OBV.
    """
    # Vérifier que les séries ont la même longueur
    if len(close) != len(volume):
        raise ValueError("Les séries des prix de clôture et des volumes doivent avoir la même longueur.")

    # Calculer les variations des prix de clôture
    delta = close.diff()

    # Déterminer le volume à ajouter ou soustraire
    obv_values = volume.where(delta > 0, -volume.where(delta < 0, 0))

    # Calculer l'OBV par sommation cumulative
    obv = obv_values.cumsum()

    return obv.fillna(0)  # Remplacer les NaN éventuels par zéro

def ma(close, window=20):
    """
    Calculer la Moyenne Mobile (MA) à partir d'une série temporelle.

    Paramètres :
        close (pd.Series) : Série temporelle des prix de clôture.
        window (int) : La période de calcul pour la MA (par défaut : 20).

    Retourne :
        pd.Series : Une série contenant les valeurs de l'EMA.
    """
    # Vérifier que la série des prix de clôture est valide
    if close.isnull().all():
        raise ValueError("La série des prix de clôture ne peut pas être vide ou contenir uniquement des NaN.")
    
    # Calculer l'EMA en utilisant la méthode ewm (Exponentially Weighted Moving)
    ma = close.rolling(window).mean()

    return ma.fillna(0) # Remplacer les NaN éventuels par zéro

def ema(close, window=20):
    """
    Calculer la Moyenne Mobile Exponentielle (EMA) à partir d'une série temporelle.

    Paramètres :
        close (pd.Series) : Série temporelle des prix de clôture.
        window (int) : La période de calcul pour l'EMA (par défaut : 20).

    Retourne :
        pd.Series : Une série contenant les valeurs de l'EMA.
    """
    # Vérifier que la série des prix de clôture est valide
    if close.isnull().all():
        raise ValueError("La série des prix de clôture ne peut pas être vide ou contenir uniquement des NaN.")

    # Calculer l'EMA en utilisant la méthode ewm (Exponentially Weighted Moving)
    ema = close.ewm(span=window, adjust=False).mean()

    return ema.fillna(0)  # Remplacer les NaN éventuels par zéro

def atr(high, low, close, window=14):
    """
    Calculer l'Average True Range (ATR) à partir de séries temporelles.

    Paramètres :
        high (pd.Series) : Série temporelle des prix les plus hauts.
        low (pd.Series) : Série temporelle des prix les plus bas.
        close (pd.Series) : Série temporelle des prix de clôture.
        window (int) : La période de calcul pour l'ATR (par défaut : 14).

    Retourne :
        pd.Series : Une série contenant les valeurs de l'ATR.
    """
    # Vérifier que toutes les séries ont la même longueur
    if not (len(high) == len(low) == len(close)):
        raise ValueError("Toutes les séries doivent avoir la même longueur.")

    # Calculer le True Range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    # Prendre le maximum entre les trois valeurs
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculer l'ATR comme la moyenne mobile exponentielle du TR
    atr = tr.ewm(span=window, adjust=False).mean()

    return atr.fillna(0)  # Remplacer les NaN éventuels par zéro

def bollinger_band_width(close, window=20, num_std=2):
    """
    Calculer la distance entre les bandes de Bollinger (Bollinger Band Width).

    Paramètres :
        close (pd.Series) : Série temporelle des prix de clôture.
        window (int) : Période de calcul pour la moyenne mobile (par défaut : 20).
        num_std (float) : Nombre d'écarts-types pour les bandes (par défaut : 2).

    Retourne :
        pd.Series : Une série contenant les valeurs de la distance entre les bandes.
    """
    # Vérifier que la série a une longueur suffisante
    if len(close) < window:
        raise ValueError("La série doit contenir au moins 'window' valeurs.")

    # Calculer la moyenne mobile simple (SMA)
    sma = close.rolling(window=window).mean()

    # Calculer l'écart-type sur la période définie
    std_dev = close.rolling(window=window).std()

    # Calculer les bandes de Bollinger
    upper_band = sma + (num_std * std_dev)
    lower_band = sma - (num_std * std_dev)

    # Calculer la distance entre les bandes
    band_width = upper_band - lower_band

    return band_width.fillna(0)  # Remplacer les NaN éventuels par zéro

def vwma(close, volume, window=20):
    """
    Calculer la Volume-Weighted Moving Average (VWMA) à partir de séries temporelles.

    Paramètres :
        close (pd.Series) : Série temporelle des prix de clôture.
        volume (pd.Series) : Série temporelle des volumes échangés.
        window (int) : La période de calcul pour la VWMA (par défaut : 20).

    Retourne :
        pd.Series : Une série contenant les valeurs de la VWMA.
    """
    # Vérifier que les séries ont la même longueur
    if len(close) != len(volume):
        raise ValueError("Les séries 'close' et 'volume' doivent avoir la même longueur.")

    # Calculer la somme glissante des (close * volume)
    weighted_price = close * volume
    sum_weighted_price = weighted_price.rolling(window=window).sum()
    sum_volume = volume.rolling(window=window).sum()

    # Calculer la VWMA
    vwma = sum_weighted_price / sum_volume

    return vwma.fillna(0)  # Remplacer les NaN éventuels par zéro

def atr(high, low, close, window=14):
    """
    Calculer l'Average True Range (ATR) à partir de séries temporelles.

    Paramètres :
        high (pd.Series) : Série temporelle des prix les plus hauts.
        low (pd.Series) : Série temporelle des prix les plus bas.
        close (pd.Series) : Série temporelle des prix de clôture.
        window (int) : La période de calcul pour l'ATR (par défaut : 14).

    Retourne :
        pd.Series : Une série contenant les valeurs de l'ATR.
    """
    # Vérifier que les séries ont la même longueur
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("Les séries 'high', 'low' et 'close' doivent avoir la même longueur.")

    # Calculer la True Range (TR)
    tr1 = high - low  # TR classique
    tr2 = (high - close.shift(1)).abs()  # TR avec le précédent close
    tr3 = (low - close.shift(1)).abs()  # TR avec le précédent close
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)  # Prendre le max des 3 TR

    # Calculer l'ATR en utilisant une moyenne mobile simple (SMA) de la True Range
    atr = true_range.rolling(window=window).mean()

    return atr.fillna(0)  # Remplacer les NaN éventuels par zéro