
import numpy as np

import torch
from torch import rand
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from xgboost import XGBRegressor

from sa.surrogate_model.base import SurrogateModel

class Regressor(nn.Module):
    def __init__(self, input_dim, mid_dim,mid_dim2=None):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.linear2 = nn.Linear(mid_dim, 1)

    def forward(self, x):
        x_ = F.relu(self.linear1(x))
        x_=self.linear2(x_)
        return x_

class ANN_set(SurrogateModel):
    '''
    Gaussian process
    '''
    def __init__(self, n_var, n_obj, **kwargs):
        super().__init__(n_var, n_obj)
        self.sms=[]
        for i in range(n_obj):
            self.sms.append([Regressor(n_var,mid_dim=random.randint(n_var//2,n_var*2)) for _ in range(3)])
        self.lr=0.05
        self.n_epoch=100

    def fit(self, X_tr, Y_tr):
        X,Y=X_tr,Y_tr
        
        for j in range(self.n_obj):
            self.sms[j]=[Regressor(self.n_var,mid_dim=random.randint(self.n_var//2,self.n_var*2)) for _ in range(3)]
        if X.shape[0]>1000:
            random_indices = np.random.choice(len(X_tr)-100, size=500, replace=False)
            X,Y=np.copy(np.concatenate([X_tr[:-100],X_tr[random_indices]]),np.concatenate([Y_tr[:-100],Y_tr[random_indices]]))
        ds_train = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())
        #dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=True)
        #criterion = torch.nn.MSELoss()
        criterion=torch.nn.MarginRankingLoss()
        for i,sm_list in enumerate(self.sms):
            for sm in sm_list:
                optimizer=optim.Adam(sm.parameters(), lr=self.lr)
                for epoch in range(self.n_epoch):
                        dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=True)
                        for j, (datapoints, labels) in enumerate(dataloader_train):
                                optimizer.zero_grad()
                                label=labels[:,i].unsqueeze(0).view(-1,1)
                                y=1.0 if label[0].item()>label[1].item() else -1.0 
                                loss=criterion(label[0].unsqueeze(0),label[1].unsqueeze(0),target=torch.tensor([y],requires_grad=True).unsqueeze(0))
                                loss.backward()
                                optimizer.step()
        #self.lr=self.lr*0.9
    def predict(self,list_module,X):
        for reg in list_module:
            reg.eval()
        preds = [regress(X).detach().numpy() for regress in list_module]
        return np.array(preds)
        
    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std
        for i,sm_list in enumerate(self.sms):
            pred=np.array([self.predict(sm_list,x)for x in torch.tensor(X).float()])
    
            y_mean=pred.mean(axis=1)
            y_mean=np.reshape(y_mean, -1)
            y_std=pred.std(axis=1)
            y_std=np.reshape(y_std, -1)


            F.append(y_mean) # y_mean: shape (N,)
            S.append(y_std)

        F = np.stack(F, axis=1)
        
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None
        
        S = np.stack(S, axis=1)
        dS = np.stack(dS, axis=1) if std and calc_gradient else None
        hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out