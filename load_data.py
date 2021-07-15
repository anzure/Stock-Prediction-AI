import mysql.connector
import numpy as np

database = mysql.connector.connect(
    host="odit-do-user-6851721-0.b.db.ondigitalocean.com",
    port=25060,
    user="odit-stock",
    password="gitj128qj9qd0oah",
    database="odit-market"
)

cursor = database.cursor()
cursor.execute("SELECT id FROM stock WHERE end_of_day=1 LIMIT 10")
cursor = cursor.fetchall()
stocks = cursor

#cursor = database.cursor()
#cursor.execute("SELECT symbol,high,low,open,close,volume,date FROM history LIMIT 10")
#cursor = cursor.fetchall()



for stock in stocks:
    stock_id = stock[0]
    cursor = database.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM history WHERE symbol='{stock_id}' LIMIT 1")
    cursor = cursor.fetchall()
    history_count = cursor[0][0]
    print(f"{stock_id} has {history_count} items.")
