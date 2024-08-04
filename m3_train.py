# import requirement libraries and tools
import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from m1_model import NeuralNetwork
from torchsummary import summary
from d3_prepareddata import get_datasets



def train(dataloader, model,optimizer,mse):
    epoch_loss = 0
    model.train()  
    
    for batch in dataloader:
        optimizer.zero_grad()          
        x,y= batch
        x = x.to("cuda")
        y = y.to("cuda")
        
        pred = model(x)
        
        loss = mse(pred[0],y)        
        loss.backward()               
        optimizer.step()      
        epoch_loss += loss.item()  
        
    return epoch_loss


def evaluate(dataloader,model,mse):
    epoch_loss = 0
    model.eval()  
    
    with torch.no_grad():
      for batch in dataloader:   
          x,y= batch
          x = x.to("cuda")
          y = y.to("cuda")
          pred = model(x)
          loss = mse(pred[0],y)              
          epoch_loss += loss.item()  
        
    return epoch_loss / len(dataloader)



def main():
    m = NeuralNetwork(4).to("cuda")
    # summary(m, (4, ))
    optimizer = optim.Adam(m.parameters())
    mse = nn.MSELoss()
    
    n_epochs = 50
    best_valid_loss = float('inf')
    train_dataloader, valid_dataloader, _, _= get_datasets()
    for epoch in range(1, n_epochs + 1):
        train_loss = train(train_dataloader,m,mse=mse,optimizer=optimizer)
        valid_loss = evaluate(valid_dataloader,m,mse=mse)
        print("train_loss>",train_loss)
        print("valid_loss>",valid_loss)
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(m, 'saved_weights.pt')
        # print("Epoch ",epoch+1)
        print(f'\tTrain Loss: {train_loss:.5f} | ' + f'\tVal Loss: {valid_loss:.5f}\n')
        
        
main()