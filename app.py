from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

app = Flask(__name__)

# Function to calculate EMA
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# Function to calculate ADX
def adx(high, low, close, length):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(window=length).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/length).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/length).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.rolling(window=length).mean()
    
    return adx

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Extracting form data
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    symbol = request.form.get('symbol')
    adx_length = int(request.form.get('adx_length'))
    ema_short = int(request.form.get('ema_short'))
    ema_long = int(request.form.get('ema_long'))
    adx_threshold = int(request.form.get('adx_threshold'))

    # Fetch historical data and process it based on the provided code
    df = yf.download(symbol, start=start_date, end=end_date)
    # Calculating indicators without talib
    df['ADX'] = adx(df['High'], df['Low'], df['Close'], adx_length)
    df['EMA_Short'] = ema(df['Close'], ema_short)
    df['EMA_Long'] = ema(df['Close'], ema_long)

    # Entry and Exit Conditions
    df['Long'] = ((df['ADX'] > adx_threshold) & (df['Close'] > df['EMA_Short']) & (df['Close'] > df['EMA_Long']))
    df['Short'] = ((df['ADX'] > adx_threshold) & (df['Close'] < df['EMA_Short']) & (df['Close'] < df['EMA_Long']))
    df['Long_Exit'] = (df['Low'].shift(1) < df['EMA_Short']) | (df['ADX'] < adx_threshold)
    df['Short_Exit'] = (df['High'].shift(1) > df['EMA_Short']) | (df['ADX'] < adx_threshold)

    # Backtesting Logic with Trade Tracking
    initial_balance = 10000.0  # Starting balance in USD
    strategy_balance = initial_balance
    position = 0  # 1 for long, -1 for short, 0 for no position
    strategy_balances = [initial_balance]
    trade_details = []
    buy_and_hold_balances = [initial_balance * (row['Close'] / df.iloc[0]['Close']) for _, row in df.iterrows()]

    for index, row in df.iterrows():
        if position == 0:
            if row['Long']:
                position = 1
                entry_price = row['Close']
                entry_date = index
            elif row['Short']:
                position = -1
                entry_price = row['Close']
                entry_date = index

        elif position == 1:
            if row['Long_Exit']:
                position = 0
                exit_price = row['Close']
                strategy_balance *= (exit_price / entry_price)
                trade_details.append({'Type': 'Long', 'Entry': entry_price, 'Exit': exit_price, 
                                    'Profit/Loss': exit_price - entry_price, 'Entry Date': entry_date, 'Exit Date': index})

        elif position == -1:
            if row['Short_Exit']:
                position = 0
                exit_price = row['Close']
                strategy_balance *= (entry_price / exit_price)
                trade_details.append({'Type': 'Short', 'Entry': entry_price, 'Exit': exit_price, 
                                    'Profit/Loss': entry_price - exit_price, 'Entry Date': entry_date, 'Exit Date': index})

        strategy_balances.append(strategy_balance)

    # Correct the length of strategy_balances to match df.index
    strategy_balances = strategy_balances[:len(df.index)]
    # Trade Summary Calculation
    total_trades = len(trade_details)
    total_profit = sum([trade['Profit/Loss'] for trade in trade_details])
    average_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
    winning_trades = len([trade for trade in trade_details if trade['Profit/Loss'] > 0])
    losing_trades = total_trades - winning_trades
    win_loss_ratio = winning_trades / losing_trades if losing_trades > 0 else float('inf')

    summary_text = (
    f"Total Trades: {total_trades}\n"
    f"Total Profit/Loss: {total_profit:.2f}\n"
    f"Average Profit/Loss per Trade: {average_profit_per_trade:.2f}\n"
    f"Winning Trades: {winning_trades}\n"
    f"Losing Trades: {losing_trades}\n"
    f"Win/Loss Ratio: {win_loss_ratio:.2f}"
    )

    # Generate the plot
    img = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, strategy_balances, label='Optimized Strategy')
    plt.plot(df.index, buy_and_hold_balances, label='Buy and Hold')
    plt.legend()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    # Prepare trade details and summary for response
    response = {
        'plot_url': plot_url,
        'trade_details': trade_details,  # This should be a list of dictionaries
        'summary': summary_text  # This should be a string
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
