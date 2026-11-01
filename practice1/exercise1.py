"""
Practice 1 - Machine Learning (BSU) — solution
===============================================
Standalone script solving all tasks of homework-practice-01.ipynb.

Section 1 (pandas): tasks 1-13, answers printed to stdout.
Section 2 (matplotlib): 4 plots saved to the `figs/` folder
(the scatter plot is also saved as a PDF, as required).

Run:  python exercise1.py
"""

import datetime
import os
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # headless backend: figures go to files, no display needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data.csv')
FIGS = os.path.join(HERE, 'figs')
os.makedirs(FIGS, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. [0.5] Open the file with the table, print the last 10 lines
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA)
print('=== 1. Last 10 rows of the table ===')
print(df.tail(10).to_string())
print('\nThe table contains %d rows and %d columns: %s.' %
      (df.shape[0], df.shape[1], ', '.join(df.columns)))

# ---------------------------------------------------------------------------
# 2. [0.25] How many orders / how many unique product categories
# ---------------------------------------------------------------------------
n_orders = df['order_id'].nunique()
n_items = df['item_name'].nunique()
print('\n=== 2 ===')
print('Orders in the sample:', n_orders)
print('Unique product categories (item_name):', n_items)

# ---------------------------------------------------------------------------
# 3. [0.25] Missing data: which columns, then fill them
# ---------------------------------------------------------------------------
print('\n=== 3. Missing values ===')
print(df.isna().sum())
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna('')
print('After filling (empty string for strings, 0 for numbers):')
print(df.isna().sum())

# ---------------------------------------------------------------------------
# 4. [0.5] item_price is a string like "$2.39 "; make a numeric column
# ---------------------------------------------------------------------------
print('\n=== 4. Numeric price ===')
print('Type of item_price:', df['item_price'].dtype)
df['price'] = df['item_price'].apply(lambda x: float(x.replace('$', '')))
print(df[['item_name', 'quantity', 'item_price', 'price']].head().to_string())
print('\nAverage price: %.2f | Minimum: %.2f | Maximum: %.2f' %
      (df['price'].mean(), df['price'].min(), df['price'].max()))
df = df.drop(columns=['item_price'])
print('Columns after removing the old price column:', list(df.columns))

# ---------------------------------------------------------------------------
# 5. [0.25] 5 cheapest and 5 most expensive products (by item_name)
# ---------------------------------------------------------------------------
# Note: in this file item_price is the *line* total (quantity is already
# included), so to compare products fairly we divide by quantity and get the
# price of one unit. Grouping by item_name also removes duplicates.
print('\n=== 5. Cheapest / most expensive products ===')
df['unit_price'] = df['price'] / df['quantity']
item_prices = df.groupby('item_name')['unit_price'].mean().sort_values()
print('5 cheapest products:')
print(item_prices.head(5).round(2).to_string())
print('\n5 most expensive products:')
print(item_prices.tail(5).round(2).to_string())

# ---------------------------------------------------------------------------
# 6. [0.5] How many times did customers buy more than 1 Chicken Bowl
# ---------------------------------------------------------------------------
chicken_multi = (df['item_name'] == 'Chicken Bowl') & (df['quantity'] > 1)
print('\n=== 6 ===')
print('Times customers bought more than 1 Chicken Bowl:', chicken_multi.sum())

# ---------------------------------------------------------------------------
# 7. [0.5] Average order bill and average number of products per order
# ---------------------------------------------------------------------------
per_order = df.groupby('order_id').agg(bill=('price', 'sum'),
                                       n_items=('quantity', 'sum'))
print('\n=== 7 ===')
print('Average order bill: $%.2f' % per_order['bill'].mean())
print('Average products per order: %.2f' % per_order['n_items'].mean())

# ---------------------------------------------------------------------------
# 8. [0.25] How many orders contained exactly 1 product
# ---------------------------------------------------------------------------
print('\n=== 8 ===')
print('Orders with exactly 1 product:', (per_order['n_items'] == 1).sum())

# ---------------------------------------------------------------------------
# 9. [0.25] Most popular product category
# ---------------------------------------------------------------------------
print('\n=== 9 ===')
top = df['item_name'].value_counts()
print('Most popular category: %s (%d purchases)' % (top.idxmax(), top.max()))

# ---------------------------------------------------------------------------
# 10. [0.5] Burrito types: most purchased / most expensive
# ---------------------------------------------------------------------------
print('\n=== 10. Burritos ===')
burritos = df[df['item_name'].str.contains('Burrito')]
burrito_stats = (burritos.groupby('item_name')
                 .agg(purchases=('item_name', 'count'),
                      max_unit_price=('unit_price', 'max'),
                      avg_unit_price=('unit_price', 'mean'))
                 .sort_values(['max_unit_price', 'avg_unit_price'],
                              ascending=False))
print(burrito_stats.round(2).to_string())
print('\nTypes of burrito:', list(burrito_stats.index))
print('Most often purchased:', burrito_stats['purchases'].idxmax())
print('Most expensive:', burrito_stats.index[0],
      '(reaches $%.2f per unit)' % burrito_stats['max_unit_price'].max())

# ---------------------------------------------------------------------------
# 11. [0.75] Orders where a product costs more than 40% of the receipt
# ---------------------------------------------------------------------------
print('\n=== 11 ===')
df['order_total'] = df.groupby('order_id')['price'].transform('sum')
big_share = df[df['price'] > 0.4 * df['order_total']]
print('Orders containing a product > 40% of the receipt:',
      big_share['order_id'].nunique())

# ---------------------------------------------------------------------------
# 12. [0.75] Diet Coke: correct $1.25 -> $1.35, compute % extra revenue
# ---------------------------------------------------------------------------
print('\n=== 12. Diet Coke correction ===')
diet_coke = (df['choice_description'].str.contains('Diet Coke', na=False)
             & (df['unit_price'] == 1.25))
before = df.loc[diet_coke, 'price'].sum()
df.loc[diet_coke, 'price'] = df.loc[diet_coke, 'quantity'] * 1.35
after = df.loc[diet_coke, 'price'].sum()
df['unit_price'] = df['price'] / df['quantity']  # keep the data consistent
print('Diet Coke revenue before the fix: $%.2f' % before)
print('Diet Coke revenue after the fix:  $%.2f' % after)
print('Percentage more money earned: %.2f%%' % ((after - before) / before * 100))

# ---------------------------------------------------------------------------
# 13. [0.75] New DataFrame from a random matrix + save to csv
# ---------------------------------------------------------------------------
print('\n=== 13 ===')
data = np.random.rand(10, 3)
new_df = pd.DataFrame(data, columns=['index', 'column1', 'column2'])
new_df = new_df.set_index('index')
print(new_df.head().round(4).to_string())
new_df.to_csv(os.path.join(HERE, 'random_matrix.csv'), index=False, header=False)
print('Saved to random_matrix.csv (no index, no column names).')

# ===========================================================================
# Section 2. Visualizations — add a random purchase date to every order
# ===========================================================================
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2018, 1, 31)
delta_seconds = int((end - start).total_seconds())

dates = pd.DataFrame(index=df.order_id.unique())
dates['date'] = [
    (start + datetime.timedelta(seconds=random.randint(0, delta_seconds)))
    .strftime('%Y-%m-%d')
    for _ in range(df.order_id.nunique())
]
df['date'] = pd.to_datetime(df.order_id.map(dates['date']))

# ---------------------------------------------------------------------------
# Viz 1. [1] Histograms: purchase amounts | average prices per item_name
# ---------------------------------------------------------------------------
order_bills = df.groupby('order_id')['price'].sum()
avg_item_prices = df.groupby('item_name')['unit_price'].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(order_bills, bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of purchase amounts')
axes[0].set_xlabel('Purchase amount, $')
axes[0].set_ylabel('Number of orders')
axes[0].grid(True, alpha=0.3)

axes[1].hist(avg_item_prices, bins=30, color='darkorange', edgecolor='white')
axes[1].set_title('Average prices of products (item_name)')
axes[1].set_xlabel('Average price of a product, $')
axes[1].set_ylabel('Number of products')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig1_histograms.png'), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Viz 2. [1] Purchase amounts vs days
# ---------------------------------------------------------------------------
daily = df.groupby('date')['price'].sum().sort_index()
plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily.values, marker='o', color='steelblue')
plt.title('Purchase amounts by day')
plt.xlabel('Date')
plt.ylabel('Total purchase amount, $')
plt.grid(True, alpha=0.3)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig2_daily_amounts.png'), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Viz 3. [1] Average purchase amounts by day of week (bar plot)
# ---------------------------------------------------------------------------
order_day = df.groupby('order_id').agg(bill=('price', 'sum'),
                                       date=('date', 'first'))
order_day['weekday'] = order_day['date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                 'Saturday', 'Sunday']
avg_by_weekday = order_day.groupby('weekday')['bill'].mean().reindex(weekday_order)

plt.figure(figsize=(9, 5))
plt.bar(avg_by_weekday.index, avg_by_weekday.values, color='seagreen')
plt.title('Average purchase amount by day of week')
plt.xlabel('Day of week')
plt.ylabel('Average purchase amount, $')
plt.ylim(0, avg_by_weekday.max() * 1.2)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig3_weekday_bars.png'), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Viz 4. [1] Money for a product vs quantity purchased (scatter) + save to PDF
# ---------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.scatter(df['quantity'], df['price'], alpha=0.6, color='crimson')
plt.title('Money spent on a product vs quantity purchased')
plt.xlabel('Quantity purchased')
plt.ylabel('Money for the product, $')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, 'fig4_scatter.png'), dpi=150)
plt.savefig(os.path.join(FIGS, 'money_vs_quantity.pdf'))  # vector PDF
plt.close()

print('\nFigures saved to %s' % FIGS)
