# import requirement libraries and tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yfinance as yf
import torch.nn as nn
import torch.functional as F
import plotly.graph_objects as go

from tqdm.notebook import tqdm
from torchsummary import summary
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv("./datasets/sh.600000.csv")
# Move column 'Close' to the first position
col_close = df.pop('close')
df.insert(0, 'close', col_close)
df.head()
df.tail()

df.shape
df.info()

df.describe().T
df.duplicated().sum()

df.plot(subplots=True, figsize=(15, 15))
plt.suptitle('stock attributes from 2016 to 2023', y=0.91)
plt.show()

df.asfreq('w', method='ffill').plot(subplots=True, figsize=(15,15), style='-')
plt.suptitle('Stock attributes over time(Weekly frequency)', y=0.91)
plt.show()


df.asfreq('m', method='ffill').plot(subplots=True, figsize=(15,15), style='-')
plt.suptitle('Stock attributes over time(Monthly frequency)', y=0.91)
plt.show()

df[['close']]



# computing moving average(ma)
ma_day = [10, 20, 50]

for ma in ma_day:
    col_name = f'MA for {ma} days'
    df[col_name] = df['close'].rolling(ma).mean()

df[['close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(figsize=(15,5))
plt.title('Comparision some MA and Close of Google stock')
plt.show()



# use pct_change to find the percent change for each day
df['Daily_Return'] = df['close'].pct_change()
# plot the daily return percentage
df.Daily_Return.plot(legend=True, figsize=(15,5))
plt.title('Daily return percentage of stock')
plt.show()

