import numpy as np
from .solver import Solver
import pygmo as pg

class All(Solver):
    '''
    Multi-objective solver
    '''
    def __init__(self,*args,**kwargs):
        super().__init__(*args, algo="all", **kwargs)
        '''
        Input:
            n_gen: number of generations to solve
            pop_init_method: method to initialize population
            algo: class of multi-objective algorithm to use
            kwargs: other keyword arguments for algorithm to initialize
        '''


    def solve(self, problem, X, Y):
        '''
        Solve the multi-objective problem
        '''
        Xp=[]
        Yp=[]
        cs=problem.real_problem.cs
        for u in np.where(~problem.real_problem.label)[0]:
            Xp.append(cs[u])
            Yp.append(problem.evaluate(cs[u]))

        Xp,Yp=np.array(Xp),np.array(Yp)    
        X_f,Y_f=self.select(Xp,Yp)
        # construct solution
        self.solution = {'x': X_f, 'y': Y_f, 'algo': 'all'}
        # fill the solution in case less than batch size
        pop_size = len(self.solution['x'])
        if pop_size < self.batch_size:
            indices = np.concatenate([np.arange(pop_size), np.random.choice(np.arange(pop_size), self.batch_size - pop_size)])
            self.solution['x'] = np.array(self.solution['x'])[indices]
            self.solution['y'] = np.array(self.solution['y'])[indices]

        return self.solution
    
    def select(self, X,Y,batch_size=100):
        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points = Y)
        ln=0
        i=0
        idx=[]
        
        while ln<batch_size:
            idx.extend(ndf[i])
            i+=1
            ln=len(idx)
        idx=idx[:batch_size]
        X_next=X[idx]
        Y_next=Y[idx]
        return np.array(X_next),np.array(Y_next)
 

    