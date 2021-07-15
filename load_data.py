import mysql.connector
import numpy as np
import pickle
import sys as system

# Connect to database
print("Connecting to database...")
database = mysql.connector.connect(
    host="odit-do-user-6851721-0.b.db.ondigitalocean.com",
    port=25060,
    user="odit-stock",
    password="gitj128qj9qd0oah",
    database="odit-market"
)
print("Connected to database.")

# Load all data
print("Loading all data...")
cursor = database.cursor()
cursor.execute("SELECT symbol,high,low,open,close,volume,date FROM history ORDER BY date ASC LIMIT 500000")
cursor = cursor.fetchall()
all_history_items = np.array(cursor)
cursor = cursor.clear()
print("Loaded all data.")

# Retrieve all tickers
print("Retrieving all tickers...")
all_tickers = []
for item in all_history_items:
    symbol = item[0]
    if not all_tickers.__contains__(symbol):
        all_tickers.append(symbol)
print("Retrieved all tickers.")

# Filter valid tickers
print("Filtering valid tickers...")
valid_tickers = []
for ticker in all_tickers:
    count = 0
    for item in all_history_items:
        symbol = item[0]
        if symbol == ticker:
            count = count + 1
    if count > 2000:
        valid_tickers.append(ticker)
print("Filtered valid tickers.")

# Filter history items
print("Filtering history items...")
valid_history_items = []
for item in all_history_items:
    symbol = item[0]
    if not valid_tickers.__contains__(symbol):
        valid_history_items.append(item)
print("Filtered history items.")

# Save training data
print("Saving dataset...")
pickle_out = open("dataset.pickle", "wb")
pickle.dump(valid_history_items, pickle_out)
pickle_out.close()
print("Saved dataset.")

# Print result
print(f"Valid history: {len(valid_history_items)}")
print(f"Total history: {len(all_history_items)}")
print(f"Valid tickers: {len(valid_tickers)}")
print(f"Total tickers: {len(all_tickers)}")
system.exit(0)
