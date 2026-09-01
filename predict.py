import json
import time
import pandas as pd
import requests

HEADERS = {"User-Agent": "FinancialAnalyzer user@example.com"}

def score_company(net_income, revenue, assets, liabilities):
    score = 0
    if revenue and revenue > 0 and net_income is not None:
        margin = net_income / revenue
        if margin > 0.20:
            score += 40
        elif margin > 0.10:
            score += 30
        elif margin > 0.05:
            score += 20
        elif margin > 0:
            score += 10
            
    if assets and assets > 0 and liabilities is not None:
        liab_ratio = liabilities / assets
        if liab_ratio < 0.4:
            score += 40
        elif liab_ratio < 0.6:
            score += 30
        elif liab_ratio < 0.8:
            score += 20
        elif liab_ratio < 1.0:
            score += 10

    if assets:
        if assets > 10_000_000_000:
            score += 20
        elif assets > 1_000_000_000:
            score += 10

    return score


def extract_latest_annual_metric(gaap_facts, metric_names):
    for tag in metric_names:
        if tag in gaap_facts and "units" in gaap_facts[tag]:
            units = gaap_facts[tag]["units"]
            for unit_key in ["USD", "pure"]:
                if unit_key in units:
                    entries = [
                        e for e in units[unit_key]
                        if e.get("form") == "10-K" and "val" in e
                    ]
                    if entries:
                        entries.sort(key=lambda x: (x.get("fy", 0), x.get("end", "")))
                        return entries[-1]["val"]
    return None


def rate_stocks_from_file(json_filepath, output_csv="rated_stocks.csv"):
    with open(json_filepath, "r") as f:
        data = json.load(f)

    # Handle both SEC Dict-of-Dicts format AND standard List format
    if isinstance(data, dict):
        stock_list = list(data.values())
    elif isinstance(data, list):
        stock_list = data
    else:
        raise ValueError("Unsupported JSON file format.")

    results = []
    total = len(stock_list)
    print(f"Loaded {total} stocks from {json_filepath}. Processing...")

    for idx, item in enumerate(stock_list):
        if not isinstance(item, dict):
            continue

        # Extract values safely across different key naming conventions
        cik_val = item.get("cik_str") or item.get("cik") or ""
        cik = str(cik_val).zfill(10)
        ticker = item.get("ticker", "UNKNOWN")
        title = item.get("title") or item.get("title_str") or item.get("entityName") or "UNKNOWN"

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        net_income, revenue, assets, liabilities = None, None, None, None
        rating_score = 0

        try:
            time.sleep(0.12)  # Respect SEC rate limits
            res = requests.get(url, headers=HEADERS, timeout=10)

            if res.status_code == 200:
                gaap = res.json().get("facts", {}).get("us-gaap", {})

                net_income = extract_latest_annual_metric(
                    gaap, ["NetIncomeLoss", "ProfitLoss"]
                )
                revenue = extract_latest_annual_metric(
                    gaap, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"]
                )
                assets = extract_latest_annual_metric(
                    gaap, ["Assets"]
                )
                liabilities = extract_latest_annual_metric(
                    gaap, ["Liabilities"]
                )

                rating_score = score_company(net_income, revenue, assets, liabilities)

        except Exception as err:
            print(f"Skipping {ticker} due to error: {err}")

        results.append({
            "Ticker": ticker,
            "Company": title,
            "CIK": cik,
            "Rating Score": rating_score,
            "Net Income": net_income,
            "Revenue": revenue,
            "Total Assets": assets,
            "Total Liabilities": liabilities,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"Progress: {idx + 1}/{total} completed.")

    df = pd.DataFrame(results)
    df = df.sort_values(by="Rating Score", ascending=False)
    df.to_csv(output_csv, index=False)
    
    print(f"Analysis complete. Results saved to '{output_csv}'.")
    return df

if __name__ == "__main__":
    rated_df = rate_stocks_from_file("company_tickers.json")