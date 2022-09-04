import numpy as np
import torch
from sa.surrogate_problem import SurrogateProblem
from sa.selection import NDS_Select
from sa.utils import Timer, find_pareto_front, calc_hypervolume
from sa.factory import init_from_config
from sa.transformation import DuplicateTransform, StandardTransform,NoTransform
import copy
'''
Main algorithm framework for Multi-Objective Bayesian Optimization
'''

class MOBO:
    '''
    Base class of algorithm framework, inherit this class with different configs to create new algorithm classes
    '''
    config = {}

    def __init__(self,config,problem, n_iter, ref_point, framework_args):
        '''
        Input:
            problem: the original / real optimization problem
            n_iter: number of iterations to optimize
            ref_point: reference point for hypervolume calculation
            framework_args: arguments to initialize each component of the framework
        '''
        self.real_problem = problem
        self.n_var, self.n_obj = problem.n_var, problem.n_obj
        self.n_iter = n_iter
        self.ref_point = ref_point
        self.config=config

        bounds = np.array([problem.xl, problem.xu])
        self.transformation = NoTransform(bounds)
        

        # framework components
        framework_args['surrogate']['n_var'] = self.n_var # for surrogate fitting
        framework_args['surrogate']['n_obj'] = self.n_obj # for surroagte fitting
        framework_args['solver']['n_obj'] = self.n_obj # for MOEA/D-EGO
        framework = init_from_config(self.config, framework_args)
        
        self.surrogate_model = framework['surrogate'] # surrogate model
        self.acquisition = framework['acquisition'] # acquisition function
        self.solver = framework['solver'] # multi-objective solver for finding the paretofront
        self.selection = framework['selection'] # selection method for choosing new (batch of) samples to evaluate on real problem
        
        # to keep track of data and pareto information (current status of algorithm)
        self.X = None
        self.Y = None
        self.sample_num = 0
        self.status = {
            'pset': None,
            'pfront': None,
            'hv': None,
            'ref_point': self.ref_point,
        }
        self.archive=[]

        # other component-specific information that needs to be stored or exported
        self.info = None

    def _update_status(self, X, Y):
        '''
        Update the status of algorithm from data
        '''
        """
            self.surrogate_model.sms[0].save_model("./models/model_bl_"+str(self.sample_num)+".json")
            self.surrogate_model.sms[1].save_model("./models/model_t_"+str(self.sample_num)+".json")
            self.archive.append({"surrb":"./models/model_bl_"+str(self.sample_num)+".json","surrt":"./models/model_t_"+str(self.sample_num)+".json","X":X,"Y":Y})
        """
        self.X_curr=X
        self.Y_curr=Y
        if self.sample_num == 0:
            self.X = X
            self.Y = Y
        else:
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])
        self.sample_num += len(X)

        self.status['pfront'], pfront_idx = find_pareto_front(self.Y, return_index=True)
        self.status['pset'] = self.X[pfront_idx]
        self.status['hv'] = calc_hypervolume(self.status['pfront'], self.ref_point)

    def solve(self, X_init, Y_init):
        '''
        Solve the real multi-objective problem from initial data (X_init, Y_init)
        '''
        # determine reference point from data if not specified by arguments
        #if self.ref_point is None:
            #self.ref_point = [-75.0,3500.0]#Y_init.max(axis=0)
        
        self.selection.set_ref_point(self.ref_point)

        self._update_status(X_init, Y_init)

        global_timer = Timer()
        print("begin the solve")
        self.it=0

        for i in range(self.n_iter):
            #print('========== Iteration %d ==========' % i)

            timer = Timer()

            print("Iteration "+str(self.it))

            # data normalization
            print("    Data transformation")
            #self.transformation.fit(X_next,Y_next)
            #X, Y = self.transformation.do(self.X_curr, self.Y_curr)
            



            # build surrogate models ( Ranking methods )
            self.surrogate_model.fit(self.X_curr,self.Y_curr)
            #timer.log('Surrogate model fitted')
            timer.log("     Surrogate model fitted")
            

            # define acquisition functions
            self.acquisition.fit(self.X, self.Y)
            timer.log('    Acq fitted fitted')

            # solve surrogate problem (Increasing population size, Tester toutes, skip solving, Mutation)
            surr_problem = SurrogateProblem(self.real_problem, self.surrogate_model, self.acquisition, self.transformation)
            solution = self.solver.solve(surr_problem, self.X_curr, self.Y_curr)
            
            timer.log('   Surrogate problem solved')

            # batch point selection
            self.selection.fit(self.X, self.Y)
            
            X_next, self.info = self.selection.select(solution, self.surrogate_model, self.status, self.transformation)
            timer.log('    Next sample batch selected')
            
            Y_next = self.real_problem.evaluate(X_next)
            #update dataset
            self._update_status(X_next, Y_next)
            timer.log('    New samples evaluated')

            # statistics
            global_timer.log('    Total runtime', reset=False)
            print('    Total evaluations: %d, hypervolume: %.4f\n' % (self.sample_num, self.status['hv']))
        
            self.it=self.it+1
            yield X_next, Y_next

    def __str__(self):
        return \
            '========== Framework Description ==========\n' + \
            f'# algorithm: {self.__class__.__name__}\n' + \
            f'# surrogate: {self.surrogate_model.__class__.__name__}\n' + \
            f'# acquisition: {self.acquisition.__class__.__name__}\n' + \
            f'# solver: {self.solver.__class__.__name__}\n' + \
            f'# selection: {self.selection.__class__.__name__}\n'