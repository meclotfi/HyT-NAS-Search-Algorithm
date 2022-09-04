
from pyexpat import model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sa.surrogate_model.base import SurrogateModel

class Regressor(nn.Module):
    def __init__(self, input_dim, mid_dim,output_dim=1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim,1)
        self.linear3 = nn.Linear(mid_dim,1)

    def forward(self, x):
        x_ =F.relu(self.linear1(x))
        x1=self.linear2(x_)
        x2=self.linear3(x_)
        return x1,x2

class ANN_Shared(SurrogateModel):
    '''
    Gaussian process
    '''
    def __init__(self, n_var, n_obj, **kwargs):
        super().__init__(n_var, n_obj)
        self.sm=Regressor(6,4)
        self.lr=0.05
        self.n_epoch=100
    
    def fit(self, X, Y):
        model=self.sm
        ds_train = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())
        dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True)
        optimizer=optim.Adam(model.parameters(), lr=self.lr)
        iteration = 0
        criterion = torch.nn.MSELoss()
        for epoch in range(self.n_epoch):
            for i, (datapoints, labels) in enumerate(dataloader_train):
                    optimizer.zero_grad()
                    y1,y2=model(datapoints)
                    loss1=criterion(y1, labels[:,0].unsqueeze(0).view(-1,1))
                    loss2=criterion(y2,labels[:,1].unsqueeze(0).view(-1,1))
                    loss=loss1+loss2
                    loss.backward()
                    optimizer.step()
    def predict(self,X):
        self.sm.eval()
        pred = self.sm(X)

        pred1,pred2=pred[0].detach().numpy()[0],pred[1].detach().numpy()[0]
        return pred1,pred2
    
    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std

        preds = np.array([self.predict(x) for x in torch.tensor(X).float()])
        
        pred1=preds[:,0]
        pred2=preds[:,1]

        F.append(pred1)
        F.append(pred2)
        F = np.stack(F, axis=1)
        
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None
        
        S = None
        dS = np.stack(dS, axis=1) if std and calc_gradient else None
        hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out