
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from xgboost import XGBRegressor


from sa.surrogate_model.base import SurrogateModel


class xgbsur(SurrogateModel):
    '''
    Gaussian process
    '''
    def __init__(self, n_var, n_obj, **kwargs):
        super().__init__(n_var, n_obj)
        self.sms=[]
        self.bs=["dart","gblinear"]
        self.n_estimators=100
        self.counter=0
        for i in range(n_obj):
            self.sms.append(XGBRegressor(booster=self.bs[i],n_estimators=self.n_estimators,random_state=42,learning_rate=1.0))

    def fit(self, X, Y):
        
        self.sms[0]=XGBRegressor(booster="dart",n_estimators=self.n_estimators)
        self.sms[1]=XGBRegressor(booster="dart",n_estimators=self.n_estimators)
        for i,sm in enumerate(self.sms):
            sm.fit(X,Y[:,i])
        self.counter+=1
    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std
        for sm in self.sms:
            preds = sm.predict(X)
            
            F.append(preds) # y_mean: shape (N,)

        F = np.stack(F, axis=1)
        
        
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None
        
        S = None
        dS = np.stack(dS, axis=1) if std and calc_gradient else None
        hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out