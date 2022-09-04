from pickletools import optimize
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.utils.optimize import _check_optimize_result
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from sympy import evaluate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sa.surrogate_model.base import SurrogateModel
from sa.utils import safe_divide

@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, 1)
        #self.blinear2 = BayesianLinear(input_dim//2, 1)
    def forward(self, x):
        x_ = self.blinear1(x)
        #x_=F.relu(x_)
        #x_=self.blinear2(x_)
        return x_


class BayesianNN(SurrogateModel):
    '''
    Gaussian process
    '''
    def __init__(self, n_var, n_obj, nu=5, **kwargs):
        super().__init__(n_var, n_obj)
        self.sms=[]
        for i in range(n_obj):
            self.sms.append(BayesianRegressor(6,1))
        self.lr=1

    def fit(self, X, Y):
        self.lr=self.lr/2
        for i, sm in enumerate(self.sms):
            ds_train = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float()[:,i])
            
            dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=2, shuffle=True)
            optimizer=optim.Adam(sm.parameters(), lr=self.lr)
            iteration = 0
            for epoch in range(10):
                for i, (datapoints, labels) in enumerate(dataloader_train):
                    
                    optimizer.zero_grad()
                    labels=torch.unsqueeze(labels, 1)
                    loss = sm.sample_elbo(inputs=datapoints,
                                    labels=labels,
                                    criterion=torch.nn.MSELoss(),
                                    sample_nbr=3)
                    loss.backward()
                    optimizer.step()
    def evaluate_regression(regress,X,y=None,samples = 10,std_multiplier = 2):
        preds = [regress(X) for _ in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        """        
        ci_upper = means + (std_multiplier * stds)
        ci_lower = means - (std_multiplier * stds)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        ic_acc = ic_acc.float().mean()
        """
        return means,stds
        
    def evaluate(self, X, std=True, calc_gradient=False, calc_hessian=False):
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std
        for sm in self.sms:
            samples=10
            preds = [sm.forward(torch.tensor(X).float()) for _ in range(samples)]
            preds = torch.stack(preds)
            y_mean = preds.mean(axis=0)
            y_mean=torch.reshape(y_mean, (-1,))
            y_std = preds.std(axis=0)
            y_std=torch.reshape(y_std, (-1,))
            
            F.append(y_mean.detach().numpy()) # y_mean: shape (N,)
            S.append(y_std.detach().numpy()) # y_std: shape (N,)

        F = np.stack(F, axis=1)
        #dF = np.stack(dF, axis=1) if calc_gradient else None
        #hF = np.stack(hF, axis=1) if calc_hessian else None
        
        S = np.stack(S, axis=1) if std else None
        #dS = np.stack(dS, axis=1) if std and calc_gradient else None
        #hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out