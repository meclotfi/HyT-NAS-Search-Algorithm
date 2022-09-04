'''
Factory for importing different components of the MOBO framework by name
'''

from sa.selection import NDS_Select
from sa.surrogate_model import ann_ens, xgbs
from sa.surrogate_model.ann import ANN
from sa.surrogate_model.ann_shared import ANN_Shared
from sa.surrogate_model.bnn import BayesianNN
from sa.surrogate_model.mix import Mix
from sa.surrogate_model.xgbr import xgbrank
from sa.surrogate_model.gaussian_process import GaussianProcess


def get_surrogate_model(name):
    from .surrogate_model import xgbsur,ANN_set,ANN_Shared,Mix,xgbrank
    
    surrogate_model = {
        'gp': GaussianProcess,
        #'ts': ThompsonSampling,
        'bnn':BayesianNN,
        "ann":ANN,
        "xgbs":xgbsur,
        #"annb":ANNB,
        "ann_set":ANN_set,
        "ann_sh":ANN_Shared,
        "mix":Mix,
        "xgbrank":xgbrank
    }

    surrogate_model['default'] = ANN_set

    return surrogate_model[name]


def get_acquisition(name):
    from .acquisition import IdentityFunc, PI, EI, UCB

    acquisition = {
        'identity': IdentityFunc,
        'pi': PI,
        'ei': EI,
        'ucb': UCB,
    }

    acquisition['default'] = IdentityFunc

    return acquisition[name]


def get_solver(name):
    from .solver import NSGA2Solver, MOEADSolver, ParEGOSolver,All

    solver = {
        'nsga2': NSGA2Solver,
        'moead': MOEADSolver,
        'parego': ParEGOSolver,
        'all':All
    }

    solver['default'] = NSGA2Solver

    return solver[name]


def get_selection(name):
    from .selection import HVI, Uncertainty, Random, DGEMOSelect, MOEADSelect,NDS_Select,NDFSelect

    selection = {
        'hvi': HVI,
        'uncertainty': Uncertainty,
        'random': Random,
        'dgemo': DGEMOSelect,
        'moead': MOEADSelect,
        'nds':NDS_Select,
        'ndf':NDFSelect
    }

    selection['default'] = HVI

    return selection[name]


def init_from_config(config, framework_args):
    '''
    Initialize each component of the MOBO framework from config
    '''
    init_func = {
        'surrogate': get_surrogate_model,
        'acquisition': get_acquisition,
        'selection': get_selection,
        'solver': get_solver,
    }

    framework = {}
    for key, func in init_func.items():
        kwargs = framework_args[key]
        if config is None:
            # no config specified, initialize from user arguments
            name = kwargs[key]
        else:
            # initialize from config specifications, if certain keys are not provided, use default settings
            name = config[key] if key in config else 'default'
        framework[key] = func(name)(**kwargs)

    return framework