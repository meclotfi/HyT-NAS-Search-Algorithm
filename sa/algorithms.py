from .mobo import MOBO

def HytSearch(Problem):
    '''
    HyTSearch
    '''
    surrogate_args={
        "surrogate":"ann_ens"
    }
    acqu_args={
        "acquisition":"ucb"
    }

    solver_args={
        "pop_init_method":"nds",
        "pop_size":100,
        "n_gen":25,
        'n_process':1,
        "batch_size":100}
    sel_args={
        "selection":'hvi',
        "batch_size":50
    }
    framework_args = {
                'surrogate':surrogate_args ,
                'acquisition': acqu_args,
                'solver': solver_args,
                'selection': sel_args,
            }
    # Initialize algorithm
    config = {
            'surrogate': 'xgbs',
            'acquisition': 'identity',
            'solver': 'nsga2',
            'selection': 'hvi',
        }
    optimizer = MOBO(config=config,problem=Problem,framework_args=framework_args,n_iter=100,ref_point=[-75.0,3500.0])
    return optimizer

def get_algorithm(name):
    '''
    Get class of algorithm by name
    '''
    algo = {
        'Hyt-Search':HytSearch
    }
    return algo[name]