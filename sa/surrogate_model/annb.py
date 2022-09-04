import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



from sa.surrogate_model.base import SurrogateModel
from xgboost import XGBRegressor

class BNNRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.blinear1 = bnn.BayesLinear(prior_mu=1, prior_sigma=0.1, in_features=input_dim//2, out_features=1)
    def forward(self, x):
        x_ = self.linear1(x)
        x_=F.relu(x_)
        out = self.blinear1(x_)
        return out


class ANNB(SurrogateModel):
    '''
    Gaussian process
    '''
    def __init__(self, n_var, n_obj, nu=5,samples=20, **kwargs):
        super().__init__(n_var, n_obj)
        self.samples=samples
        self.sms=[]
        for i in range(n_obj):
            self.sms.append(BNNRegressor(6,1))
        self.lr=0.1

    def fit(self, X, Y):
        self.lr=0.1
        for i, sm in enumerate(self.sms):
            model=sm
            ds_train = torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float()[:,i])
            dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True)
            mse_loss = nn.MSELoss()
            kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
            kl_weight = 0.02
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(100):
                for i, (datapoints, labels) in enumerate(dataloader_train):
                            pre = model(datapoints)
                            labels=labels.unsqueeze(1)
                            mse = mse_loss(pre, labels)
                            kl = kl_loss(model)
                            cost = mse + kl_weight*kl
                            optimizer.zero_grad()
                            cost.backward()
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
            preds = [sm.forward(torch.tensor(X).float()) for _ in range(self.samples)]
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