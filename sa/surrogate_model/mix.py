import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from xgboost import XGBRegressor

from sa.surrogate_model.base import SurrogateModel

class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.linear2 = nn.Linear(input_dim//2, 1)
    def forward(self, x):
        x_ = self.linear1(x)
        x_=F.relu(x_)
        x_=self.linear2(x_)
        return x_

class Mix(SurrogateModel):
    def __init__(self, n_var, n_obj, **kwargs):
        super().__init__(n_var, n_obj)
        self.sms=[]
        
        self.sms.append(XGBRegressor(
        ))
        self.sms.append(Regressor(6,1))
        
        
        self.lr=0.01
        self.n_epoch=200
    
    def fit(self,X,Y):
        self.fitA(X,Y)
        self.fitB(X,Y)

    def fitA(self, X, Y):
        i=0
        sm=self.sms[0]
        sm.fit(X,Y[:,i])

    def fitB(self, X, Y):
        ds_train = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())
        dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=4, shuffle=True)
        criterion = torch.nn.MSELoss()
        i=1
        sm=self.sms[1]
        optimizer=optim.Adamax(sm.parameters(), lr=self.lr)
        for epoch in range(self.n_epoch):
                    for j, (datapoints, labels) in enumerate(dataloader_train):
                            optimizer.zero_grad()
                            label=labels[:,i].unsqueeze(0).view(-1,1)
                            loss=criterion(sm(datapoints), label)
                            loss.backward()
                            optimizer.step()
    def evaluate_regression(regress,X,y=None,samples = 10,std_multiplier = 2):
        preds = [regress(X) for _ in range(samples)]
        return preds
        
    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std
        preds=[]
        sm=self.sms[0]
        preds = sm.predict(X)
        F.append(preds) # y_mean: shape (N,)


        sm=self.sms[1]
        preds = np.array([sm(x).detach().numpy()[0] for x in torch.tensor(X).float()])
        F.append(preds) # y_mean: shape (N,)

        F = np.stack(F, axis=1)
        
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None
        
        S = None
        dS = np.stack(dS, axis=1) if std and calc_gradient else None
        hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out