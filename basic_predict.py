import pandas as pd
import yfinance as yf
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bs4 import BeautifulSoup
import requests

# 1. Initialize FinBERT globally
print("Loading FinBERT sentiment engine...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def fetch_sentiment(ticker):
    """Scrapes financial news and runs FinBERT sentiment analysis."""
    url = f"https://finance.yahoo.com/quote/{ticker}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            text = " ".join([p.get_text() for p in soup.find_all(['p', 'h3'])])[:512]
            
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return {
                "positive": round(probs[0][0].item(), 2),
                "negative": round(probs[0][1].item(), 2),
                "neutral": round(probs[0][2].item(), 2)
            }
    except Exception:
        pass
    return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

def fetch_market_factors(ticker_symbol):
    """Fetches technical momentum and fundamental metrics."""
    try:
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="60d")
        if hist.empty:
            return None
            
        # 14-Day RSI Calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = round(100 - (100 / (1 + rs.iloc[-1])), 2)
        
        info = stock.info
        pe_ratio = info.get('forwardPE', None)
        inst_ownership = info.get('heldPercentInstitutions', 0)
        
        return {
            "rsi": rsi,
            "pe_ratio": pe_ratio,
            "institutional_pct": round(inst_ownership * 100, 2) if inst_ownership else "N/A"
        }
    except Exception:
        return None

def generate_user_action_prediction(sentiment, factors):
    """
    Evaluates multi-factor conditions and outputs an actionable recommendation.
    """
    rsi = factors['rsi']
    neg_sent = sentiment['negative']
    pos_sent = sentiment['positive']
    
    # Decision Matrix logic
    if neg_sent > 0.45 and rsi < 30:
        action = "🟢 BUY (Oversold Dip-Buy Setup)"
        reason = "News is negative, but RSI indicates extreme oversold conditions. High likelihood of institutional floor forming."
    elif pos_sent > 0.55 and rsi < 65:
        action = "🟢 BUY (Momentum Continuation)"
        reason = "Strong positive sentiment paired with healthy technical momentum."
    elif pos_sent > 0.55 and rsi >= 70:
        action = "🟡 HOLD / TAKE PROFIT (Overbought Warning)"
        reason = "Sentiment is bullish, but stock is overbought (RSI > 70). Risk of short-term pullback."
    elif neg_sent > 0.45 and rsi >= 35:
        action = "🔴 SELL / AVOID (Sustained Downtrend)"
        reason = "Negative text sentiment combined with weak technical momentum."
    else:
        action = "🟡 HOLD (Neutral / Wait for Signal)"
        reason = "Mixed signals across sentiment and technical metrics."
        
    return action, reason

def run_watchlist_assistant():
    print("=== AI Stock Watchlist & Prediction Assistant ===")
    user_input = input("Enter ticker symbols for your watchlist (e.g., PYPL, AMD, AAPL): ")
    watchlist = [t.strip().upper() for t in user_input.split(",") if t.strip()]
    
    if not watchlist:
        print("No tickers entered.")
        return

    for ticker in watchlist:
        print(f"\n==========================================")
        print(f"📊 REPORT FOR WATCHLIST ITEM: {ticker}")
        print(f"==========================================")
        
        sentiment = fetch_sentiment(ticker)
        factors = fetch_market_factors(ticker)
        
        if factors:
            print(f"• FinBERT Sentiment   : Positive {sentiment['positive']} | Negative {sentiment['negative']}")
            print(f"• 14-Day RSI         : {factors['rsi']}")
            print(f"• Institutional Hold : {factors['institutional_pct']}%")
            
            action, reason = generate_user_action_prediction(sentiment, factors)
            
            print(f"\n🔮 MODEL PREDICTION / ACTION:")
            print(f"   Recommended Action : {action}")
            print(f"   Underlying Logic   : {reason}")
        else:
            print(f"Unable to fetch complete data for {ticker}.")

if __name__ == "__main__":
    run_watchlist_assistant()