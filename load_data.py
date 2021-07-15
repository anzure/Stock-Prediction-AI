import mysql.connector
import numpy as np
import sys as system

# Connect to database
database = mysql.connector.connect(
    host="odit-do-user-6851721-0.b.db.ondigitalocean.com",
    port=25060,
    user="odit-stock",
    password="gitj128qj9qd0oah",
    database="odit-market"
)

# Load all data
cursor = database.cursor()
cursor.execute("SELECT symbol,high,low,open,close,volume,date FROM history LIMIT 500000")
cursor = cursor.fetchall()
history_items = np.array(cursor)
cursor = cursor.clear()

# Retrieve all tickers
all_tickers = []
for item in history_items:
    symbol = item[0]
    if not all_tickers.__contains__(symbol):
        all_tickers.append(symbol)

# Filter valid tickers
valid_tickers = []
for ticker in all_tickers:
    count = 0
    for item in history_items:
        symbol = item[0]
        if symbol == ticker:
            count = count + 1
    if count > 2000:
        valid_tickers.append(ticker)

# Print result
print(f"Valid tickers: {len(valid_tickers)}")
print(f"Total tickers: {len(all_tickers)}")
system.exit(0)
