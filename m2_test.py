
# # import requirement libraries and tools
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.optim as optim
# import yfinance as yf
# import torch.nn as nn
# import torch.functional as F
# import plotly.graph_objects as go

# from tqdm.notebook import tqdm

# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import TensorDataset, DataLoader



# # def test():
# #   model=torch.load('saved_weights.pt')
# #   x_test= torch.tensor(x_test).float()
# #   with torch.no_grad():
# #     y_test_pred = model(x_test)
# #   y_test_pred = y_test_pred.numpy()[0]

# #   idx=0
# #   plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
# #           y_test[:,idx], color='black', label='test target')

# #   plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test_pred.shape[0]),
# #           y_test_pred[:,idx], color='green', label='test prediction')

# #   plt.title('future stock prices')
# #   plt.xlabel('time [days]')
# #   plt.ylabel('normalized price')
# #   plt.legend(loc='best')


# #   index_values = df[len(df) - len(y_test):].index
# #   col_values = ['Open', 'Low', 'High', 'Close']
# #   df_results = pd.DataFrame(data=y_test_pred, index=index_values, columns=col_values)



# #   # Create a trace for the candlestick chart
# #   candlestick_trace = go.Candlestick(
# #       x=df_results.index,
# #       open=df_results['Open'],
# #       high=df_results['High'],
# #       low=df_results['Low'],
# #       close=df_results['Close'],
# #       name='Candlestick'
# #   )

# #   # Create the layout
# #   layout = go.Layout(
# #       title='GOOG Candlestick Chart',
# #       xaxis=dict(title='Date'),
# #       yaxis=dict(title='Price', rangemode='normal')
# #   )

# #   # Create the figure and add the candlestick trace and layout
# #   fig = go.Figure(data=[candlestick_trace], layout=layout)

# #   # Update the layout of the figure
# #   fig.update_layout(xaxis_rangeslider_visible=False)

# #   # Show the figure
# #   fig.show()