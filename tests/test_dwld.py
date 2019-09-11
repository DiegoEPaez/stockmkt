from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import datetime

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2002, 1, 2)

df = pdr.get_data_yahoo('^GSPC', start, end)
df.index.names = ['DATE']
df = df.rename(columns={'Adj Close':'SPX'})
print(df)
