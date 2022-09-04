from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler
'''
Data transformations (normalizations) for fitting surrogate model
'''


### 1-dim scaler

class Scaler(ABC):
    
    def fit(self, X):
        return self

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def inverse_transform(self, X):
        pass


class BoundedScaler(Scaler):
    '''
    Scale data to [0, 1] according to bounds
    '''
    def __init__(self, bounds):
        self.bounds = bounds

    def transform(self, X):
        return np.clip((X - self.bounds[0]) / (self.bounds[1] - self.bounds[0]), 0, 1)

    def inverse_transform(self, X):
        return np.clip(X, 0, 1) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]


### 2-dim transformation

class Transformation:

    def __init__(self, x_scaler, y_scaler):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def fit(self, x, y):
        self.x_scaler = self.x_scaler.fit(x)
        self.y_scaler = self.y_scaler.fit(y)

    def do(self, x=None, y=None):
        assert x is not None or y is not None
        if x is not None:
            x_res = self.x_scaler.transform(np.atleast_2d(x))
            if len(np.array(x).shape) < 2:
                x_res = x_res.squeeze()

        if y is not None:
            y_res = self.y_scaler.transform(np.atleast_2d(y))
            if len(np.array(y).shape) < 2:
                y_res = y_res.squeeze()

        if x is not None and y is not None:
            return x_res, y_res
        elif x is not None:
            return x_res
        elif y is not None:
            return y_res

    def undo(self, x=None, y=None):
        assert x is not None or y is not None
        if x is not None:
            x_res = self.x_scaler.inverse_transform(np.atleast_2d(x))
            if len(np.array(x).shape) < 2:
                x_res = x_res.squeeze()

        if y is not None:
            y_res = self.y_scaler.inverse_transform(np.atleast_2d(y))
            if len(np.array(y).shape) < 2:
                y_res = y_res.squeeze()

        if x is not None and y is not None:
            return x_res, y_res
        elif x is not None:
            return x_res
        elif y is not None:
            return y_res
    

class StandardTransform(Transformation):

    def __init__(self, x_bound,ymin=[-30,0],ymax=[0,1000]):
        super().__init__(
            x_scaler=BoundedScaler(x_bound),
            y_scaler=MinMaxScaler()
        )
        print([ymin,ymax])
        self.y_scaler.fit(np.array([ymin,ymax]))

class NoTransform(Transformation):

    def __init__(self, x_bound):
        super().__init__(
            BoundedScaler(x_bound),
            StandardScaler()
        )
    def do(self, x=None, y=None):
        if x is not None and y is not None:
            return x, y
        elif x is not None:
            return x
        elif y is not None:
            return y
    def undo(self, x=None, y=None):
        if x is not None and y is not None:
            return x, y
        elif x is not None:
            return x
        elif y is not None:
            return y

transform_lst_so = [lambda x: math.log(x/1000,2)/5,
                 lambda x: (x-1)/3,
                 lambda x: (math.log(x, 2)-8)/2,
                 lambda x: (x-1024)/1024,
                 lambda x: (x-8)/8,
                 lambda x: (x-0.0003)/0.0007]

transform_lst_sw = [lambda x: math.log(x/1000,2)/5,
                 lambda x: (x-1)/5,
                 lambda x: (math.log(x, 2)-8)/2,
                 lambda x: (x-1024)/1024,
                 lambda x: (x-8)/8,
                 lambda x: (x-0.0003)/0.0007]

class DuplicateTransform(Transformation):

    def __init__(self,dataset,mt_dataset="so-en"):
        self.bench=dataset
        self.cs_str=self.bench.get_configuration_space()
        self.cs = []
        transform_lst=[]
        if mt_dataset=="so-en":
                transform_lst=transform_lst_so
        else:
                print("sw-en")
                transform_lst=transform_lst_sw
        for c in self.cs_str:
            cf = [float(f) for f in c.split('\t')]
            self.cs.append([transform_lst[i](cf[i]) for i in range(len(cf))])
        self.label = np.full(len(self.cs), False)
        self.cs = np.array(self.cs)
        
    def do(self, x=None, y=None):
        x_res=[]
        y_res=[]
        assert x is not None or y is not None
        if x is not None:
            if(len(np.array(x).shape)<2):
                x_res=self._replace_neighbor(np.array(x))
            else:
                for i in range(x.shape[0]):
                    x_i=self._replace_neighbor(np.array(x[i]))
                    x_res.append(x_i)
                    y_res.append(y[i])
        return np.array(x_res),np.array(y_res)


    def undo(self, x=None, y=None):
        if y is None:
            if x is not None:
                return x
        else:
            if x is not None:
                return x,y
            else:
                return y
    
    def _replace_neighbor(self, x):
        dis = np.full(len(self.cs), np.inf)
        for u in np.where(~self.label)[0]:
            dis[u] = spatial.distance.cosine(x, self.cs[u])
        id = np.nanargmin(dis)
        if id in np.where(self.label)[0]:
            for u in np.where(~self.label)[0]:
                dis[u] = spatial.distance.euclidean(x, self.cs[u])
            id = np.nanargmin(dis)
        self.label[id]=True
        print(id)
        return self.cs[id]

    def fit(self, x, y):
        pass

def round_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor
class Transform_Space(Transformation):
    def __init__(self):
        
        self.transformss=[
                 lambda x: ((x-1)/4),lambda x,y: (x-1)/4,lambda x,y: (x-1)/4,lambda x,y: (x-1)/4,
                 lambda x,y: (math.log(x, 2))/3,lambda x,y: (math.log(x, 2))/3,lambda x,y: (math.log(x, 2))/3,lambda x,y: (math.log(x, 2))/3,lambda x,y: (math.log(x, 2))/3,
                 lambda x,y: (x-8)/32,lambda x,y: (2*(x/y)-2)/3,lambda x,y: (2*(x/y)-2)/3,lambda x,y: (2*(x/y)-2)/3,lambda x,y: (2*(x/y)-2)/3,
                 lambda x,y: (math.log(x, 2))/3,lambda x,y: (math.log(x, 2))/3,lambda x,y: (math.log(x, 2))/3,
                 lambda x,y: (2*(x/self.highestPowerof2(y))-2)/3,lambda x,y: (2*(x/self.highestPowerof2(y))-2)/3,lambda x,y: (2*(x/self.highestPowerof2(y))-2)/3,
                 lambda x,y: (math.log(x, 2)-1)/3,lambda x,y: (math.log(x, 2)-1)/3,lambda x,y: (math.log(x, 2)-1)/3
                 ]
        self.inv_transformss=[
                lambda x: int((x*4)+1),lambda x,y: int((x*4)+1),lambda x,y: int((x*4)+1),lambda x,y: int((x*4)+1),
                lambda x,y: 2**(int(x*3)),lambda x,y: 2**(int(x*3)),lambda x,y: 2**(int(x*3)),lambda x,y: 2**(int(x*3)),lambda x,y: 2**(int(x*3)),
                lambda x,y: (((x*32)+8)//8)*8,lambda x,y: (int(x*3+2)/2)*y,lambda x,y: (int(x*3+2)/2)*y,lambda x,y: (int(x*3+2)/2)*y,lambda x,y: (int(x*3+2)/2)*y,
                lambda x,y: 2**(int(x*3)),lambda x,y: 2**(int(x*3)),lambda x,y: 2**(int(x*3)),
                lambda x,y: (int(x*3+2)/2)*self.highestPowerof2(y),lambda x,y: (int(x*3+2)/2)*self.highestPowerof2(y),lambda x,y: (int(x*3+2)/2)*self.highestPowerof2(y),
                lambda x,y: 2**(int(x*3)+1),lambda x,y: 2**(int(x*3)+1),lambda x,y: 2**(int(x*3)+1)
                           ]
    
    def highestPowerof2(self,n):
        res = 0
        n=int(n)
        for i in range(n, 0, -1):
            
            # If i is a power of 2
            if ((i & (i - 1)) == 0):
                res = i
                break
        return res


    def do(self, x=None, y=None):
        x_res=[]
        assert x is not None or y is not None
        if x is not None:
            for c in x:
                x_res.append([round_up(self.transformss[0](c[0]),2)]+[round_up(self.transformss[i](c[i],c[i-1]),2) for i in range(1,17)]+[round_up(self.transformss[i](c[i],c[i-6]),2) for i in range(17,len(c))])
            return np.array(x_res)


    def undo(self, x=None, y=None):
        x_res=[]
        assert x is not None or y is not None
        for c in x:
            l1=[self.inv_transformss[0](c[0])]
            l2=[]
            prec=0
            for i in range(1,17):
                prec=int(self.inv_transformss[i](c[i],prec))
                l2.append(prec)
            l3=[int(self.inv_transformss[i](c[i],l2[i-7])) for i in range(17,len(c))]
            x_res.append(l1+l2+l3)
            
        return np.array(x_res)