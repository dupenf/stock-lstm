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



def get_datasets(batch_size=32, shuffle=False):
    df = pd.read_csv("./datasets/sh.600000.csv")

    # normalize data
    df2 = df.copy(deep=True)
    scaler = MinMaxScaler(feature_range=(0,15)).fit(df2.low.values.reshape(-1,1))
    df2['open'] = scaler.transform(df2.open.values.reshape(-1,1))
    df2['high'] = scaler.transform(df2.high.values.reshape(-1,1))
    df2['low'] = scaler.transform(df2.low.values.reshape(-1,1))
    df2['close'] = scaler.transform(df2.close.values.reshape(-1,1))    
    df2.to_csv("./datasets/features.csv")
    data = df2[['open','high','low', 'close']].values
    
    # divide the entire dataset into three parts. 80% for the training set, 10% for the validation set and the remaining 10% for the test set:
    seq_len=11  # 11 day 
    sequences=[]
    for index in range(len(data) - seq_len + 1): 
        sequences.append(data[index: index + seq_len])
    sequences= np.array(sequences)

    valid_set_size_percentage = 10 
    test_set_size_percentage = 10 
    
    valid_set_size = int(np.round(valid_set_size_percentage/100*sequences.shape[0]))  
    test_set_size  = int(np.round(test_set_size_percentage/100*sequences.shape[0]))
    train_set_size = sequences.shape[0] - (valid_set_size + test_set_size)
    
    x_train = sequences[:train_set_size,:-1,:]
    y_train = sequences[:train_set_size,-1,:]
        
    x_valid = sequences[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = sequences[train_set_size:train_set_size+valid_set_size,-1,:]
        
    # 剩下的都是test set
    x_test = sequences[train_set_size+valid_set_size:,:-1,:]
    y_test = sequences[train_set_size+valid_set_size:,-1,:]

    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train).float()

    x_valid = torch.tensor(x_valid).float()
    y_valid = torch.tensor(y_valid).float()

    train_dataset = TensorDataset(x_train,y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    valid_dataset = TensorDataset(x_valid,y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle)


    return train_dataloader, valid_dataloader, sequences, scaler


# get_datasets()