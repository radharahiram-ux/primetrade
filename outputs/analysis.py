import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

OUT = '/home/claude/primetrade_analysis/outputs/'
SENTIMENT_ORDER = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
SENTIMENT_NUM   = {'Extreme Fear': 1, 'Fear': 2, 'Neutral': 3, 'Greed': 4, 'Extreme Greed': 5}
PALETTE = {
    'Extreme Fear': '#d62728',
    'Fear':         '#ff7f0e',
    'Neutral':      '#7f7f7f',
    'Greed':        '#2ca02c',
    'Extreme Greed':'#1f77b4',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.size':        11,
})

# ─── STEP 1: LOAD & CLEAN ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — LOADING & CLEANING DATA")
print("=" * 60)

fg = pd.read_csv('/mnt/user-data/uploads/fear_greed_index.csv')
fg['date'] = pd.to_datetime(fg['date'])
fg = fg[['date', 'value', 'classification']].rename(columns={'classification': 'Classification'})
fg['Classification'] = fg['Classification'].str.strip()
print(f"Fear & Greed Index: {fg.shape[0]} rows, {fg['date'].min().date()} to {fg['date'].max().date()}")
print(fg['Classification'].value_counts())

df = pd.read_csv('/mnt/user-data/uploads/historical_data__1_.csv')
df.columns = df.columns.str.strip()
print(f"\nTrader Data: {df.shape[0]} rows, {df.shape[1]} cols")

# Parse date
df['date'] = pd.to_datetime(df['Timestamp IST'], dayfirst=True, errors='coerce').dt.normalize()
df = df.dropna(subset=['date'])
df['Closed PnL'] = pd.to_numeric(df['Closed PnL'], errors='coerce').fillna(0)
df['Size USD']   = pd.to_numeric(df['Size USD'],   errors='coerce').fillna(0)
print(f"Trader date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Unique traders: {df['Account'].nunique()}")
print(f"Unique coins: {df['Coin'].nunique()}")

# ─── STEP 2: MERGE ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — MERGING DATASETS")
print("=" * 60)

merged = df.merge(fg[['date', 'Classification', 'value']], on='date', how='inner')
merged = merged[merged['Classification'].isin(SENTIMENT_ORDER)]
print(f"Merged rows: {merged.shape[0]}")
print(f"Coverage by sentiment:\n{merged['Classification'].value_counts()}")

# ─── STEP 3: SENTIMENT REGIME ANALYSIS ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — SENTIMENT REGIME ANALYSIS")
print("=" * 60)

merged['win'] = (merged['Closed PnL'] > 0).astype(int)

sentiment_stats = merged.groupby('Classification').agg(
    Total_Trades       = ('Closed PnL', 'count'),
    Total_PnL          = ('Closed PnL', 'sum'),
    Avg_PnL_per_Trade  = ('Closed PnL', 'mean'),
    Win_Rate_pct       = ('win', lambda x: round(x.mean() * 100, 2)),
    Avg_Trade_Size_USD = ('Size USD', 'mean'),
    Median_PnL         = ('Closed PnL', 'median'),
).reindex(SENTIMENT_ORDER).round(4)

print(sentiment_stats.to_string())
sentiment_stats.to_csv(OUT + 'sentiment_analysis.csv')
print("\nSaved: sentiment_analysis.csv")

# ─── STEP 4: TRADER PERFORMANCE RANKING ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — TRADER PERFORMANCE RANKING")
print("=" * 60)

trader_stats = merged.groupby('Account').agg(
    Total_PnL    = ('Closed PnL', 'sum'),
    Total_Trades = ('Closed PnL', 'count'),
    Win_Rate_pct = ('win', lambda x: round(x.mean() * 100, 2)),
    Avg_Trade_USD= ('Size USD', 'mean'),
).round(4)

best_regime = merged.groupby(['Account', 'Classification'])['Closed PnL'].mean().reset_index()
best_regime = best_regime.loc[best_regime.groupby('Account')['Closed PnL'].idxmax()]
best_regime = best_regime.rename(columns={'Classification': 'Best_Regime', 'Closed PnL': 'Best_Regime_AvgPnL'})
trader_stats = trader_stats.merge(best_regime[['Account', 'Best_Regime']], on='Account', how='left')
trader_stats = trader_stats.sort_values('Total_PnL', ascending=False).reset_index()
trader_stats['Rank'] = trader_stats.index + 1

print("TOP 10 TRADERS:")
print(trader_stats.head(10)[['Rank','Account','Total_PnL','Total_Trades','Win_Rate_pct','Best_Regime']].to_string(index=False))
print("\nBOTTOM 10 TRADERS:")
print(trader_stats.tail(10)[['Rank','Account','Total_PnL','Total_Trades','Win_Rate_pct','Best_Regime']].to_string(index=False))

trader_stats.to_csv(OUT + 'trader_rankings.csv', index=False)
print("\nSaved: trader_rankings.csv")

# ─── STEP 5: CONTRARIAN & MOMENTUM TRADERS ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — CONTRARIAN & MOMENTUM TRADERS")
print("=" * 60)

regime_trader = merged.groupby(['Account', 'Classification']).agg(
    Avg_PnL  = ('Closed PnL', 'mean'),
    Trades   = ('Closed PnL', 'count'),
    Win_Rate = ('win', lambda x: round(x.mean() * 100, 2)),
).reset_index()

contrarian = regime_trader[
    (regime_trader['Classification'] == 'Extreme Fear') & (regime_trader['Trades'] >= 5)
].sort_values('Avg_PnL', ascending=False).head(5)
print("TOP 5 CONTRARIAN TRADERS (best in Extreme Fear):")
print(contrarian[['Account','Avg_PnL','Trades','Win_Rate']].to_string(index=False))

momentum = regime_trader[
    regime_trader['Classification'].isin(['Greed', 'Extreme Greed'])
].groupby('Account').agg(Avg_PnL=('Avg_PnL','mean'), Trades=('Trades','sum')).reset_index()
momentum = momentum[momentum['Trades'] >= 5].sort_values('Avg_PnL', ascending=False).head(5)
print("\nTOP 5 MOMENTUM TRADERS (best in Greed/Extreme Greed):")
print(momentum[['Account','Avg_PnL','Trades']].to_string(index=False))

# ─── STEP 6: CORRELATION ANALYSIS ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — CORRELATION ANALYSIS")
print("=" * 60)

merged['Sentiment_Score'] = merged['Classification'].map(SENTIMENT_NUM)
corr_data = merged[['Sentiment_Score', 'Closed PnL', 'win', 'Size USD']].copy()
corr_data.columns = ['Sentiment_Score', 'Closed_PnL', 'Win_Flag', 'Trade_Size_USD']
corr_matrix = corr_data.corr()
print(corr_matrix.round(4).to_string())

# ─── STEP 7: VISUALIZATIONS ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — GENERATING VISUALIZATIONS")
print("=" * 60)

# 1. Avg PnL by sentiment
fig, ax = plt.subplots(figsize=(10, 5))
vals = sentiment_stats['Avg_PnL_per_Trade'].reindex(SENTIMENT_ORDER)
colors = [PALETTE[s] for s in SENTIMENT_ORDER]
bars = ax.bar(SENTIMENT_ORDER, vals, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if v >= 0 else -0.03),
            f'${v:.2f}', ha='center', va='bottom', fontsize=9)
ax.set_title('Average PnL per Trade by Market Sentiment', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Sentiment Regime')
ax.set_ylabel('Average Closed PnL (USD)')
plt.tight_layout()
plt.savefig(OUT + 'plot1_avg_pnl_by_sentiment.png', dpi=150)
plt.close()
print("Saved plot1")

# 2. Win rate by sentiment
fig, ax = plt.subplots(figsize=(10, 5))
wr = sentiment_stats['Win_Rate_pct'].reindex(SENTIMENT_ORDER)
bars = ax.bar(SENTIMENT_ORDER, wr, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(50, color='gray', linewidth=0.8, linestyle='--', label='50% baseline')
for bar, v in zip(bars, wr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
ax.set_title('Win Rate by Market Sentiment', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Sentiment Regime')
ax.set_ylabel('Win Rate (%)')
ax.legend()
plt.tight_layout()
plt.savefig(OUT + 'plot2_win_rate_by_sentiment.png', dpi=150)
plt.close()
print("Saved plot2")

# 3. Number of trades by sentiment
fig, ax = plt.subplots(figsize=(10, 5))
tc = sentiment_stats['Total_Trades'].reindex(SENTIMENT_ORDER)
bars = ax.bar(SENTIMENT_ORDER, tc, color=colors, edgecolor='white', linewidth=0.5)
for bar, v in zip(bars, tc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{int(v):,}', ha='center', va='bottom', fontsize=9)
ax.set_title('Number of Trades by Market Sentiment', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Sentiment Regime')
ax.set_ylabel('Total Trades')
plt.tight_layout()
plt.savefig(OUT + 'plot3_trade_count_by_sentiment.png', dpi=150)
plt.close()
print("Saved plot3")

# 4. PnL distribution box plot
fig, ax = plt.subplots(figsize=(12, 6))
data_by_regime = [merged[merged['Classification'] == s]['Closed PnL'].clip(-500, 500).values
                  for s in SENTIMENT_ORDER]
bp = ax.boxplot(data_by_regime, labels=SENTIMENT_ORDER, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='.', markersize=2, alpha=0.3))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('PnL Distribution by Market Sentiment (clipped ±$500)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Sentiment Regime')
ax.set_ylabel('Closed PnL (USD)')
plt.tight_layout()
plt.savefig(OUT + 'plot4_pnl_distribution_boxplot.png', dpi=150)
plt.close()
print("Saved plot4")

# 5. Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 6))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = False
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Correlation Matrix — Sentiment vs Trade Metrics', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(OUT + 'plot5_correlation_heatmap.png', dpi=150)
plt.close()
print("Saved plot5")

# 6. Total PnL over time by sentiment
daily_pnl = merged.groupby(['date', 'Classification'])['Closed PnL'].sum().reset_index()
daily_total = daily_pnl.groupby('date')['Closed PnL'].sum().reset_index()
daily_sentiment = daily_pnl.sort_values('Closed PnL', ascending=False).drop_duplicates('date')[['date','Classification']]
daily_total = daily_total.merge(daily_sentiment, on='date', how='left')
daily_total = daily_total.sort_values('date')
daily_total['Cumulative_PnL'] = daily_total['Closed PnL'].cumsum()

fig, ax = plt.subplots(figsize=(14, 5))
for regime in SENTIMENT_ORDER:
    subset = daily_total[daily_total['Classification'] == regime]
    ax.scatter(subset['date'], subset['Cumulative_PnL'],
               color=PALETTE[regime], label=regime, s=8, alpha=0.7)
ax.plot(daily_total['date'], daily_total['Cumulative_PnL'],
        color='black', linewidth=0.8, alpha=0.4)
ax.set_title('Cumulative PnL Over Time (colored by dominant sentiment)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative PnL (USD)')
ax.legend(loc='upper left', markerscale=2, fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT + 'plot6_cumulative_pnl_over_time.png', dpi=150)
plt.close()
print("Saved plot6")

# 7. Top 10 traders by total PnL
fig, ax = plt.subplots(figsize=(12, 6))
top10 = trader_stats.head(10)
short_labels = [a[:10] + '…' for a in top10['Account']]
bar_colors = [PALETTE.get(r, '#7f7f7f') for r in top10['Best_Regime']]
bars = ax.barh(short_labels[::-1], top10['Total_PnL'][::-1], color=bar_colors[::-1], edgecolor='white')
for bar, v in zip(bars, top10['Total_PnL'][::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'${v:,.0f}', va='center', fontsize=9)
ax.set_title('Top 10 Traders by Total PnL', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Total Closed PnL (USD)')
handles = [plt.Rectangle((0,0),1,1, color=PALETTE[s]) for s in SENTIMENT_ORDER]
ax.legend(handles, SENTIMENT_ORDER, title='Best Regime', loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(OUT + 'plot7_top10_traders.png', dpi=150)
plt.close()
print("Saved plot7")

# ─── STEP 8: KEY INSIGHTS ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

best_pnl_regime = sentiment_stats['Avg_PnL_per_Trade'].idxmax()
best_wr_regime  = sentiment_stats['Win_Rate_pct'].idxmax()
worst_regime    = sentiment_stats['Avg_PnL_per_Trade'].idxmin()
total_pnl_all   = merged['Closed PnL'].sum()
overall_wr      = round(merged['win'].mean() * 100, 2)
most_active     = sentiment_stats['Total_Trades'].idxmax()

