import numpy as np
from problems.lhs import lhs
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Solver:
    '''
    Multi-objective solver
    '''
    def __init__(self,algo, n_gen=10, pop_init_method=FloatRandomSampling(), batch_size=10, **kwargs):
        '''
        Input:
            n_gen: number of generations to solve
            pop_init_method: method to initialize population
            algo: class of multi-objective algorithm to use
            kwargs: other keyword arguments for algorithm to initialize
        '''
        self.n_gen = n_gen
        self.pop_init_method = pop_init_method
        self.batch_size = batch_size
        self.algo_type = algo
        self.algo_kwargs = kwargs
        self.solution = None

    def solve(self, problem, X, Y):
        '''
        Solve the multi-objective problem
        '''
        # initialize population
        sampling = self._get_sampling(X, Y)

        # setup algorithm
        algo = self.algo_type(sampling=sampling,**self.algo_kwargs)

        # optimization
        res = minimize(problem, algo, ('n_gen', self.n_gen))
        
        Xp=res.pop.get('X')
        Yp=res.pop.get('F')
        #X_f,Y_f=self._pareto(Xp,Yp)
        Xf,Yf=Xp,Yp
        # construct solution
        self.solution = {'x': Xf, 'y': Yf, 'algo': res.algorithm}

        # fill the solution in case less than batch size
        pop_size = len(self.solution['x'])
        if pop_size < self.batch_size:
            indices = np.concatenate([np.arange(pop_size), np.random.choice(np.arange(pop_size), self.batch_size - pop_size)])
            self.solution['x'] = np.array(self.solution['x'])[indices]
            self.solution['y'] = np.array(self.solution['y'])[indices]
        
        
        return self.solution

    def _get_sampling(self, X, Y):
        '''
        Initialize population from data
        '''
        
        if self.pop_init_method == 'lhs':
            sampling = LatinHypercubeSampling()
        elif self.pop_init_method == 'nds':
            sorted_indices = NonDominatedSorting().do(Y)
            pop_size = self.algo_kwargs['pop_size']
            sampling = X[np.concatenate(sorted_indices)][:pop_size]
            # NOTE: use lhs if current samples are not enough
            if len(sampling) < pop_size:
                rest_sampling = lhs(X.shape[1], pop_size - len(sampling))
                sampling = np.vstack([sampling, rest_sampling])
        elif self.pop_init_method == 'random':
            sampling = FloatRandomSampling()
        else:
            sampling = FloatRandomSampling()

        return sampling
    def _pareto(self,X,Y):
        fronts=[]
        front_labels=[False for _ in range(Y.shape[0])]
        for i in range(Y.shape[0]):
              to_remove=[]
              add_fronts=True
              y=Y[i]
              j=0
              l=len(fronts)
              # Loop over cy
              while(j<l):
                f=fronts[j]
                if ((y[0]>f[0])&(y[1]>f[1])):
                  add_fronts=False
                if (y[0]<f[0])&(y[1]<f[1]):
                  fronts.pop(j)
                  front_labels[j]=False
                  l-=1
                  j-=1
                j+=1
              if(add_fronts):
                  fronts.append(y)
                  front_labels[i]=True
        return X[front_labels],Y[front_labels]
                  

