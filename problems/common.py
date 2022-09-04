from hashlib import new
import numpy as np
from pymoo.factory import get_from_list, get_reference_directions
from problems import *
from sklearn.cluster import KMeans,AffinityPropagation
from scipy import interpolate
from pymoo.factory import get_sampling
from pymoo.interface import sample

from sa.transformation import Transform_Space
from .lhs import lhs


def generate_vector_space():
    # number of blocks 0,3
    poss_nb=[1,2,3,4]
    nb=np.random.choice(poss_nb,4)
    
    # exp_ratio 4,8
    poss_exr=[1,2,4]
    exr=np.random.choice(poss_exr,5)
    
    #Out channel 9,13
    poss_c=[8,16,24,32]
    Oc1=np.random.choice(poss_c,1)[0]
    Ocs=[Oc1]
    Oc=Oc1
    poss_ex=[1,1.5,2]
    for i in range(4):
        Oc=np.random.choice(poss_ex,1)[0]*Oc
        Ocs.append(Oc)

    #N heads 14,16
    poss_nh=[1,2,4]
    nh=np.random.choice(poss_nh,3)
   

    #ffn_ratio 17,19
    poss_fnr=[1,1.5,2]
    fnr=np.random.choice(poss_fnr,3)

     #patches 20,22
    poss_patch=[2,4,8]
    patch=np.random.choice(poss_patch,3)

    return np.concatenate((nb,exr,Ocs,nh,fnr,patch))
def generate_initial_samples_SS(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
   

    # NOTE: when it's really hard to get feasible samples, the program hangs here
    tr=Transform_Space()
    X = tr.do([generate_vector_space() for _  in range(n_sample)])
    Y = problem.evaluate(X, return_values_of=['F'])
    return X,Y

def generate_initial_samples_beta(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
    X_feasible = np.zeros((0, problem.n_var))
    Y_feasible = np.zeros((0, problem.n_obj))

    # NOTE: when it's really hard to get feasible samples, the program hangs here
    X = np.random.beta(a=.5,b=.5,size=(n_sample,problem.n_var))
    Y = problem.evaluate(X, return_values_of=['F'])

    return X, Y
def generate_initial_samples_clustering(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
    #clustering = KMeans(n_clusters=n_sample).fit(problem.cs)
    clustering = AffinityPropagation().fit(problem.cs)
    X_init = clustering.cluster_centers_
    #indices = np.random.permutation(np.arange(len(X_init)))[:n_sample]
    X = X_init
    Y = problem.evaluate(X, return_values_of=['F'])
    #interp = interpolate.NearestNDInterpolator(X,Y)
    #xnew = np.random.beta(a=.5,b=.5,size=(100,problem.n_var))
    #ynew = interp(xnew)
    #X,Y=np.concatenate([X,xnew]),np.concatenate([Y,ynew])
    return X, Y
def generate_initial_samples_lhs(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
    #clustering = KMeans(n_clusters=n_sample).fit(problem.cs)
    
    X = lhs(n=6, samples=n_sample, criterion="c")
    #indices = np.random.permutation(np.arange(len(X_init)))[:n_sample]
    Y = problem.evaluate(X, return_values_of=['F'])
    #interp = interpolate.NearestNDInterpolator(X,Y)
    #xnew = np.random.beta(a=.5,b=.5,size=(100,problem.n_var))
    #ynew = interp(xnew)
    #X,Y=np.concatenate([X,xnew]),np.concatenate([Y,ynew])
    return X, Y

def generate_initial_samples(problem, n_sample):
    '''
    Generate feasible initial samples.
    Input:
        problem: the optimization problem
        n_sample: number of initial samples
    Output:
        X, Y: initial samples (design parameters, performances)
    '''
    X_feasible = np.zeros((0, problem.n_var))
    Y_feasible = np.zeros((0, problem.n_obj))

    # NOTE: when it's really hard to get feasible samples, the program hangs here
    while len(X_feasible) < n_sample:
        X = lhs(problem.n_var, n_sample)
        X = problem.xl + X * (problem.xu - problem.xl)
        Y, feasible = problem.evaluate(X, return_values_of=['F', 'feasible'])
        feasible = feasible.flatten()
        X_feasible = np.vstack([X_feasible, X[feasible]])
        Y_feasible = np.vstack([Y_feasible, Y[feasible]])
    
    indices = np.random.permutation(np.arange(len(X_feasible)))[:n_sample]
    X, Y = X_feasible[indices], Y_feasible[indices]
    return X, Y
