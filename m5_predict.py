

# import requirement libraries and tools
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from m1_model import NeuralNetwork
from torchsummary import summary
from d3_prepareddata import get_datasets
import pandas as pd

def predict():
    
    model=torch.load('saved_weights.pt').to("cuda")
    _, _, sequences,scaler = get_datasets()
    # Get the last sequence of historical data as features for predicting the next 10 days
    last_sequence = sequences[-1:, 1:, :]
    print(last_sequence)
    last_sequence = torch.from_numpy(last_sequence).float()
    

    # Generate predictions for the next 10 days
    PRED_DAYS = 10
    with torch.no_grad():
        for i in range(PRED_DAYS):
            last_sequence = last_sequence.to("cuda")
            pred_i = model(last_sequence)
            last_sequence = torch.cat((last_sequence, pred_i), dim=1)
            last_sequence = last_sequence[:, 1:, :]


    last_sequence = last_sequence.to("cpu")
    pred_days = last_sequence.reshape(PRED_DAYS, 4).numpy()

    # inverse transform the predicted values
    pred_days = scaler.inverse_transform(pred_days)

    df_pred = pd.DataFrame(
        data=pred_days,
        columns=['open', 'high', 'low', 'close']
    )

    print(df_pred)
    
predict()