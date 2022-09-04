
from tokenize import group
import numpy as np
from xgboost import XGBRanker
from sa.surrogate_model.base import SurrogateModel


class xgbrank(SurrogateModel):
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
            self.sms.append(XGBRanker(n_estimators=100,booster='gbtree',objective='rank:ndcg',random_state=42,learning_rate=0.1))

    def fit(self, X, Y):
        
        
        
        """
        self.counter+=1
        if(self.counter>6 and self.counter<12):
            nb=(self.counter-6)*10+30
            X,Y=X[-nb:,:],Y[-nb:,:]
        if(self.counter>11):
            nb=(self.counter-10)*10+20
            X,Y=X[-nb:,:],Y[-nb:,:]
        print(X.shape)
        """
        #if (X.shape[0]>100):
        #   X,Y=X[-100:,:],Y[-100:,:]

        #print(val)
        X_train,Y_train=np.copy(X),np.copy(Y)
        #if (self.counter>=9):
         #       random_indices = np.random.choice(len(X), size=int(len(X)*0.7), replace=False)
          #      X_train,Y_train=np.copy(X[random_indices]),np.copy(Y[random_indices])
        #   lr=0.1
        self.sms[0]=XGBRanker(n_estimators=1000,booster=self.bs[0],objective='rank:ndcg',random_state=7)
        self.sms[1]=XGBRanker(n_estimators=1000,booster=self.bs[1],objective='rank:ndcg',random_state=7)
        for i,sm in enumerate(self.sms):
            sm.fit(X_train,Y_train[:,i],group=[X_train.shape[0]])#qid=np.array([1 for _ in range(X_train.shape[0])]))
        self.counter+=1
    def evaluate(self, X, std=False, calc_gradient=False, calc_hessian=False):
        F, dF, hF = [], [], [] # mean
        S, dS, hS = [], [], [] # std
        for sm in self.sms:
            preds = - sm.predict(X)
            
            F.append(preds) # y_mean: shape (N,)

        F = np.stack(F, axis=1)
        
        
        dF = np.stack(dF, axis=1) if calc_gradient else None
        hF = np.stack(hF, axis=1) if calc_hessian else None
        
        S = None
        dS = np.stack(dS, axis=1) if std and calc_gradient else None
        hS = np.stack(hS, axis=1) if std and calc_hessian else None

        out = {'F': F, 'dF': dF, 'hF': hF, 'S': S, 'dS': dS, 'hS': hS}
        return out