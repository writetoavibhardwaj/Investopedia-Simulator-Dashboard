
from __future__ import annotations
from typing import List, Optional
from datetime import timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="Portfolio — Performance & Day-wise (Final)", layout="wide")


def _normalize_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df

def _clean_money_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r'[\\$,]', '', regex=True).str.strip(), errors='coerce')

@st.cache_data(show_spinner=False)
def read_any(uploaded):
    if uploaded is None:
        return None
    # try csv then excel (xlsx/xls)
    try:
        return pd.read_csv(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
            return pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return None

def standardize_performance(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _normalize_cols(df)
    if df is None or df.empty or 'date' not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'], errors='coerce').dt.tz_localize(None)
    out = out.dropna(subset=['date'])

    if 'stock_value' in out.columns:
        out['portfolio_value'] = _clean_money_series(out['stock_value'])
    elif 'portfolio_value' in out.columns:
        out['portfolio_value'] = _clean_money_series(out['portfolio_value'])

    if 'cash' in out.columns:
        out['cash'] = _clean_money_series(out['cash'])

    # account total: prefer 'account_value', otherwise cash+portfolio
    if 'account_value' in out.columns:
        out['account_value'] = _clean_money_series(out['account_value'])
    else:
        out['account_value'] = out.get('cash', 0.0) + out.get('portfolio_value', 0.0)

    keep = [c for c in ['date','account_value','portfolio_value','cash'] if c in out.columns]
    out = out[keep].sort_values('date').reset_index(drop=True)

    # percent vs first day
    for col in ['account_value','portfolio_value','cash']:
        if col in out.columns and not out[col].isna().all():
            base = out[col].iloc[0]
            out[f'{col}_pct'] = np.where(base > 0, out[col]/base - 1.0, np.nan)
    return out

# -------------------- TRADES --------------------
def standardize_trades(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _normalize_cols(df)
    if df is None or df.empty:
        return pd.DataFrame()

    # aliases tailored to your file: Date, Symbol, Trade Type, Quantity, Price
    cand = {
        "ticker":  ["ticker","symbol","sec","security"],
        "date":    ["date","trade_date","datetime","timestamp"],
        "quantity":["quantity","qty","shares","units","filled","executed_quantity","signed_quantity"],
        "price":   ["price","avg_price","trade_price","execution_price","fill_price","average_price"],
        "action":  ["action","side","type","transaction_type","buy_sell","trade_type"],
        "fees":    ["fees","fee","commission","fees_and_commissions"],
        "amount":  ["amount","proceeds","net_amount","cash_amount","gross_amount","net_cash"],
    }
    def pick(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        return None

    col_date   = pick(cand["date"])
    col_ticker = pick(cand["ticker"])
    col_qty    = pick(cand["quantity"])
    col_price  = pick(cand["price"])
    col_action = pick(cand["action"])
    col_fees   = pick(cand["fees"])
    col_amount = pick(cand["amount"])

    if not col_date or not col_qty or not col_price or not col_ticker:
        return pd.DataFrame()

    out = df.copy()
    out[col_date] = pd.to_datetime(out[col_date], errors='coerce').dt.tz_localize(None)
    out = out.dropna(subset=[col_date])
    out['ticker']   = out[col_ticker].astype(str).str.upper().str.strip()
    out['quantity'] = pd.to_numeric(out[col_qty], errors='coerce').fillna(0.0).abs()
    out['price']    = _clean_money_series(out[col_price]).fillna(0.0)
    out['fees']     = _clean_money_series(out[col_fees]) if col_fees else 0.0
    out['amount']   = _clean_money_series(out[col_amount]) if col_amount else np.nan

    if col_action:
        act_raw = out[col_action].astype(str).str.lower()
        out['action'] = np.where(act_raw.str.contains('sell|sold|short'), 'SELL', 'BUY')
    else:
        out['action'] = 'BUY'

    std = out.rename(columns={col_date:'date'})
    std = std[['date','ticker','action','quantity','price','fees']].sort_values(['date','ticker']).reset_index(drop=True)
    return std

def moving_avg_ledger(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=['date','ticker','action','quantity','price','fees','run_qty','run_avg_cost','cash_flow','notional'])
    rows = []
    for tkr, g in trades.groupby('ticker'):
        qty = 0.0; avg = 0.0
        for _, r in g.sort_values('date').iterrows():
            action = str(r.get('action','')).upper().strip()
            q = float(r['quantity']); p = float(r['price']); f = float(r['fees'])
            signed = q if action.startswith('B') else -abs(q)
            notional = q * p
            if signed >= 0:  # BUY
                new_qty = qty + signed
                if new_qty <= 1e-12:
                    avg = 0.0; qty = 0.0
                else:
                    avg = (qty*avg + signed*p) / new_qty
                    qty = new_qty
                cash_flow = -(signed*p + f)
            else:            # SELL
                sell_qty = abs(signed)
                cash_flow = sell_qty*p - f
                qty = max(qty - sell_qty, 0.0)
            rows.append({
                'date': r['date'], 'ticker': tkr, 'action': action,
                'quantity': q, 'price': p, 'fees': f,
                'run_qty': qty, 'run_avg_cost': avg,
                'cash_flow': cash_flow, 'notional': notional if action=='BUY' else -notional
            })
    return pd.DataFrame(rows).sort_values(['date','ticker']).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not yf or not tickers:
        return pd.DataFrame(columns=['date','ticker','close'])
    data = yf.download(
        tickers=list(sorted(set(tickers))),
        start=(start - timedelta(days=3)).date(),
        end=(end + timedelta(days=3)).date(),
        auto_adjust=True, threads=True, progress=False
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close'].reset_index().melt(id_vars=['Date'], var_name='ticker', value_name='close').rename(columns={'Date':'date'})
    else:
        data = data.reset_index().rename(columns={'Date':'date','Close':'close'})
        data['ticker'] = tickers[0]
    data['date'] = pd.to_datetime(data['date']).dt.tz_localize(None)
    return data.dropna(subset=['close'])

def daily_positions(ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=['date','ticker','qty','avg_cost'])
    dates = pd.date_range(start.date(), end.date(), freq='D')
    frames = []
    for tkr, g in ledger.groupby('ticker'):
        g = g[['date','run_qty','run_avg_cost']].rename(columns={'run_qty':'qty','run_avg_cost':'avg_cost'})
        g = g.sort_values('date').drop_duplicates('date', keep='last').set_index('date')
        g = g.reindex(dates).ffill().fillna({'qty':0.0,'avg_cost':0.0})
        g['date'] = g.index; g['ticker'] = tkr
        frames.append(g.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)

def daily_cash_from_ledger(ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, start_cash: float) -> pd.DataFrame:
    dates = pd.date_range(start.date(), end.date(), freq='D')
    cf = ledger.groupby('date', as_index=False)['cash_flow'].sum() if not ledger.empty else pd.DataFrame({'date':[], 'cash_flow':[]})
    s = pd.DataFrame({'date': dates})
    s = s.merge(cf, on='date', how='left').fillna({'cash_flow':0.0})
    s['cash'] = start_cash + s['cash_flow'].cumsum()
    return s

st.sidebar.title("Uploads")
perf_file   = st.sidebar.file_uploader("Performance history (.csv/.xlsx/.xls)", type=["csv","xlsx","xls"])
trades_file = st.sidebar.file_uploader("Trade history (.csv/.xlsx/.xls)", type=["csv","xlsx","xls"])
start_cash  = float(st.sidebar.number_input("Start Cash (for trades)", value=0.0, step=1000.0))
st.sidebar.markdown("---")

st.header("Performance — Amount & % (from performance file)")
perf_raw = read_any(perf_file) if perf_file is not None else None
perf = standardize_performance(perf_raw) if perf_raw is not None else pd.DataFrame()

if perf.empty:
    st.info("Upload a performance file to see Account / Portfolio / Cash charts.")
else:
    st.subheader("Amount (USD)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(px.line(perf, x='date', y='account_value', title="Account Value"), use_container_width=True)
    with c2:
        if 'portfolio_value' in perf.columns:
            st.plotly_chart(px.line(perf, x='date', y='portfolio_value', title="Portfolio Value (Stock)"), use_container_width=True)
        else:
            st.info("No Portfolio/Stock Value column in file.")
    with c3:
        if 'cash' in perf.columns:
            st.plotly_chart(px.line(perf, x='date', y='cash', title="Cash Position"), use_container_width=True)
        else:
            st.info("No Cash column in file.")

    # Percent charts
    st.subheader("Percent (vs first day)")
    p1, p2, p3 = st.columns(3)
    with p1:
        fig = px.line(perf, x='date', y='account_value_pct', title="Account Value %")
        fig.update_yaxes(tickformat='.1%'); st.plotly_chart(fig, use_container_width=True)
    with p2:
        if 'portfolio_value_pct' in perf.columns:
            fig = px.line(perf, x='date', y='portfolio_value_pct', title="Portfolio Value %")
            fig.update_yaxes(tickformat='.1%'); st.plotly_chart(fig, use_container_width=True)
    with p3:
        if 'cash_pct' in perf.columns:
            fig = px.line(perf, x='date', y='cash_pct', title="Cash %")
            fig.update_yaxes(tickformat='.1%'); st.plotly_chart(fig, use_container_width=True)

st.header("Day-wise Portfolio — Buys/Sells, Positions, % Weights (from trades)")
trades_raw = read_any(trades_file) if trades_file is not None else None
trades = standardize_trades(trades_raw) if trades_raw is not None else pd.DataFrame()

if trades.empty:
    st.info("Upload a trades file to build daily positions and weights.")
else:
    start_date = trades['date'].min()
    end_date   = max(trades['date'].max(), pd.Timestamp.today().normalize())

    ledger = moving_avg_ledger(trades)
    act = ledger.copy()
    act['notional'] = act['quantity'] * act['price']
    day_activity = act.groupby(['date','ticker','action'], as_index=False).agg(
        qty=('quantity','sum'),
        avg_price=('price','mean'),
        fees=('fees','sum'),
        notional=('notional','sum'),
        cash_flow=('cash_flow','sum')
    ).sort_values(['date','ticker'])

    st.subheader("Daily Buys/Sells by Ticker")
    st.dataframe(day_activity, use_container_width=True, hide_index=True)

    # Daily positions
    pos_daily = daily_positions(ledger, start_date, end_date)
    tickers = sorted(trades['ticker'].unique().tolist())
    if 'PORTF' in tickers:
        tickers.remove('PORTF')
    prices = fetch_prices(tickers, start_date, end_date) if tickers else pd.DataFrame(columns=['date','ticker','close'])

    if not prices.empty and not pos_daily.empty:
        merged = pos_daily.merge(prices, on=['date','ticker'], how='left')
        merged['market_value'] = merged['qty'] * merged['close']
    else:
        merged = pos_daily.copy()
        merged['close'] = np.nan
        merged['market_value'] = np.nan

    day_mv = merged.groupby('date', as_index=False)['market_value'].sum().rename(columns={'market_value':'day_market_value'})
    merged = merged.merge(day_mv, on='date', how='left')
    merged['weight'] = np.where(merged['day_market_value']>0, merged['market_value']/merged['day_market_value'], np.nan)

    st.subheader("Ticker % of Portfolio Over Time")
    if merged['weight'].notna().any():
        fig_alloc = px.area(merged, x='date', y='weight', color='ticker', groupnorm='fraction',
                            title="Ticker Weights (Stacked %)")
        fig_alloc.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_alloc, use_container_width=True)
    else:
        st.info("Need daily prices to compute weights for your tickers.")

    st.subheader("Per-Day Breakdown: Ticker % of Portfolio")
    pick_date = st.date_input("Pick a date", value=pd.to_datetime(end_date).date(),
                              min_value=pd.to_datetime(start_date).date(),
                              max_value=pd.to_datetime(end_date).date())
    day_sel = merged[merged['date'] == pd.to_datetime(pick_date)].copy()
    if not day_sel.empty and day_sel['weight'].notna().any():
        day_sel = day_sel.sort_values('weight', ascending=False)
        fig_day = px.bar(day_sel, x='ticker', y='weight', title=f"Weights on {pick_date}")
        fig_day.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_day, use_container_width=True)
        st.dataframe(day_sel[['ticker','qty','avg_cost','close','market_value','weight']],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No priced positions for the selected date.")
