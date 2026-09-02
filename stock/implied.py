import os
import requests
import torch
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yfinance as yf

app = Flask(__name__)

print("Loading FinBERT NLP model into memory...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


def analyze_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    pos = round(probs[0][0].item() * 100, 1)
    neg = round(probs[0][1].item() * 100, 1)
    neu = round(probs[0][2].item() * 100, 1)
    
    return {
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "confidence": max(pos, neg, neu)
    }


def fetch_implied_volatility(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        if not expirations:
            return None
        
        nearest_exp = expirations[0]
        opt_chain = ticker.option_chain(nearest_exp)
        calls = opt_chain.calls
        current_price = ticker.fast_info['lastPrice']
        
        atm_calls = calls[
            (calls['strike'] >= current_price * 0.95) & 
            (calls['strike'] <= current_price * 1.05)
        ]
        
        if atm_calls.empty:
            avg_iv = calls['impliedVolatility'].mean()
        else:
            avg_iv = atm_calls['impliedVolatility'].mean()
            
        return round(avg_iv * 100, 2)
    except Exception:
        return None


def fetch_text_data(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all(['p', 'h3'])
            text = " ".join([p.get_text() for p in paragraphs])
            return text if len(text) > 200 else None
    except Exception:
        return None


def evaluate_combined_signal(sentiment, iv):
    pos = sentiment['positive'] / 100
    neg = sentiment['negative'] / 100
    
    iv_status = "High Volatility Expected" if iv and iv > 40 else "Normal/Low Volatility"
    
    if pos > 0.55 and iv and iv > 40:
        signal = "🟢 High-Conviction Bullish (High Sentiment + Expected Volatility)"
    elif pos > 0.55:
        signal = "🟢 Mildly Bullish (Positive Sentiment, Low Volatility)"
    elif neg > 0.45 and iv and iv > 40:
        signal = "🔴 High-Conviction Bearish (Negative Sentiment + Expected Volatility)"
    elif neg > 0.45:
        signal = "🔴 Mildly Bearish (Negative Sentiment, Low Volatility)"
    else:
        signal = "🟡 Neutral / Wait for Catalyst"
        
    return signal, iv_status


def generate_stock_forecast(ticker_symbol, sentiment=None, iv=None, days_ahead=30):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="6mo")
        if df.empty:
            return None

        dates = [d.strftime('%Y-%m-%d') for d in df.index]
        prices = [round(p, 2) for p in df['Close'].tolist()]
        current_price = prices[-1]

        recent_prices = prices[-30:] if len(prices) >= 30 else prices
        x = np.arange(len(recent_prices))
        hist_slope, _ = np.polyfit(x, recent_prices, 1)
        hist_daily_return = hist_slope / current_price

        if sentiment and iv:
            pos_score = sentiment.get("positive", 0) / 100.0
            neg_score = sentiment.get("negative", 0) / 100.0
            sentiment_bias = pos_score - neg_score

            daily_iv = (iv / 100.0) / np.sqrt(252)
            sentiment_daily_return = sentiment_bias * daily_iv * 0.4
            blended_daily_return = (0.2 * hist_daily_return) + (0.8 * sentiment_daily_return)
        else:
            blended_daily_return = hist_daily_return

        last_date = pd.to_datetime(dates[-1])
        future_dates = [(last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_ahead + 1)]

        predicted_prices = []
        simulated_price = current_price
        for _ in range(days_ahead):
            simulated_price *= (1 + blended_daily_return)
            predicted_prices.append(round(simulated_price, 2))

        return {
            "historical_dates": dates,
            "historical_prices": prices,
            "forecast_dates": future_dates,
            "forecast_prices": predicted_prices,
            "current_price": current_price
        }
    except Exception as e:
        print(f"Error generating forecast for {ticker_symbol}: {e}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    tickers = data.get("tickers", [])

    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    results = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue

        text = fetch_text_data(ticker)
        iv = fetch_implied_volatility(ticker)

        if text:
            sentiment = analyze_sentiment(text[:512])
            signal, iv_status = evaluate_combined_signal(sentiment, iv)
            chart_data = generate_stock_forecast(ticker, sentiment=sentiment, iv=iv)

            if chart_data:
                results.append({
                    "ticker": ticker,
                    "sentiment": sentiment,
                    "iv": iv if iv is not None else "N/A",
                    "iv_status": iv_status,
                    "signal": signal,
                    "chart": chart_data,
                    "status": "success"
                })
                continue

        results.append({
            "ticker": ticker,
            "status": "error",
            "message": f"Could not retrieve complete market data or news for {ticker}."
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)