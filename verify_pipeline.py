import sqlite3

conn = sqlite3.connect('crypto_data.db')
cur = conn.cursor()
print(cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall())
for table in ['sourcecode_metrics','network_metrics','economics_metrics','sentiment_metrics','accessibility_metrics']:
    print(table, cur.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0])
conn.close()
