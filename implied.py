from bs4 import BeautifulSoup
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yfinance as yf

# 1. Load FinBERT Model globally
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text):
    """Generates FinBERT sentiment probabilities."""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return {
        "positive": round(probs[0][0].item(), 2),
        "negative": round(probs[0][1].item(), 2),
        "neutral": round(probs[0][2].item(), 2)
    }

def fetch_implied_volatility(ticker_symbol):
    """
    Fetches option chain data and returns average IV 
    for options expiring soonest near current price.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        
        if not expirations:
            return None
        
        # Target the nearest expiration date
        nearest_exp = expirations[0]
        opt_chain = ticker.option_chain(nearest_exp)
        calls = opt_chain.calls
        
        # Calculate current stock price
        current_price = ticker.fast_info['lastPrice']
        
        # Filter for Near-the-Money options (+/- 5% of stock price)
        atm_calls = calls[
            (calls['strike'] >= current_price * 0.95) & 
            (calls['strike'] <= current_price * 1.05)
        ]
        
        if atm_calls.empty:
            avg_iv = calls['impliedVolatility'].mean()
        else:
            avg_iv = atm_calls['impliedVolatility'].mean()
            
        return round(avg_iv * 100, 2)  # Convert to percentage
    except Exception as e:
        print(f"Error fetching IV for {ticker_symbol}: {e}")
        return None

def fetch_text_data(ticker):
    """Fetches text snippets for sentiment analysis."""
    url = f"https://finance.yahoo.com/quote/{ticker}/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all(['p', 'h3'])
            text = " ".join([p.get_text() for p in paragraphs])
            return text if len(text) > 200 else None
    except Exception:
        return None

def evaluate_combined_signal(sentiment, iv):
    """
    Pairs Sentiment (Direction) with IV (Magnitude) for a multi-factor signal.
    """
    pos = sentiment['positive']
    neg = sentiment['negative']
    
    # Categorize IV
    iv_status = "High Volatility Expected" if iv and iv > 40 else "Normal/Low Volatility"
    
    # Generate multi-factor signal
    if pos > 0.55 and iv and iv > 40:
        signal = "🟢 High-Conviction Bullish (High Sentiment + Market Expects Big Move)"
    elif pos > 0.55:
        signal = "🟢 Mildly Bullish (Positive Tone, Low Volatility Expected)"
    elif neg > 0.45 and iv and iv > 40:
        signal = "🔴 High-Conviction Bearish (Negative Tone + Market Expects Big Move)"
    elif neg > 0.45:
        signal = "🔴 Mildly Bearish (Negative Tone, Low Volatility Expected)"
    else:
        signal = "🟡 Neutral / Wait for Catalyst"
        
    return signal, iv_status

def run_watchlist_analyzer():
    print("=== Multi-Factor Stock Sentiment & IV Analyzer ===")
    
    # Prompt the user to input a list of stock tickers
    user_input = input("Enter ticker symbols separated by commas (e.g. AMD, AAPL, NVDA): ")
    
    # Clean the input into a list of uppercase tickers
    watchlist = [ticker.strip().upper() for ticker in user_input.split(",") if ticker.strip()]
    
    if not watchlist:
        print("No valid tickers entered.")
        return

    print(f"\nProcessing watchlist: {', '.join(watchlist)}\n")
    
    for ticker_symbol in watchlist:
        print(f"--- Analyzing {ticker_symbol} ---")
        text = fetch_text_data(ticker_symbol)
        iv = fetch_implied_volatility(ticker_symbol)
        
        if text:
            sentiment = analyze_sentiment(text[:512])
            signal, iv_status = evaluate_combined_signal(sentiment, iv)
            
            print(f"Implied Volatility (IV): {iv}% -> ({iv_status})")
            print(f"Sentiment Breakdown    : {sentiment}")
            print(f"Combined Signal        : {signal}\n")
        else:
            print(f"Could not retrieve text content for {ticker_symbol}.\n")

# --- RUNNING THE ANALYZER ---
if __name__ == "__main__":
    test_ticker = run_watchlist_analyzer()
    print(f"\n--- Analyzing Multi-Factor Data for {test_ticker} ---")
    
    text = fetch_text_data(test_ticker)
    iv = fetch_implied_volatility(test_ticker)
    
    if text:
        sentiment = analyze_sentiment(text[:512])
        signal, iv_status = evaluate_combined_signal(sentiment, iv)
        
        print(f"Implied Volatility (IV): {iv}% -> ({iv_status})")
        print(f"Sentiment Breakdown    : {sentiment}")
        print(f"Combined Signal        : {signal}")
    else:
        print("Could not retrieve text content.")