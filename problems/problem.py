import numpy as np
import autograd
from autograd.numpy import row_stack
from pymoo.core.problem import Problem as PymooProblem
from pymoo.util.misc import at_least_2d_array
import pandas as pd
from xgboost import XGBRegressor
from sa.transformation import Transform_Space


'''
Problem definition built upon Pymoo's Problem class, added some custom features
'''

class Problem(PymooProblem):

    def __init__(self,dataset=None,n_var=23,n_obj=2,xl=0,xu=1):
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
        
        # construct or load the surrogate models
        tr=Transform_Space()
        self.acc_surr=XGBRegressor(booster="gblinear",n_estimators=500)
        data=pd.read_csv("./datasets/models_acc_pr.csv",header=None)
        X,Y=tr.do(data.drop(columns=[23]).values),data[23].values
        self.acc_surr.fit(X,Y)


        self.lat_surr=XGBRegressor(booster="dart",n_estimators=200)
        data=pd.read_csv("./datasets/models_lat_pre.csv",header=None)
        X,Y=tr.do(data.drop(columns=[23]).values),data[23].values
        self.lat_surr.fit(X,Y)
        
        # train 
        self.elementwise_evaluation=True
        self.archive=[]
    
    def evaluate(self,
                 X,
                 *args,
                 return_values_of="auto",
                 return_as_dictionary=False,
                 **kwargs):

        """
        Evaluate the given problem.

        The function values set as defined in the function.
        The constraint values are meant to be positive if infeasible. A higher positive values means "more" infeasible".
        If they are 0 or negative, they will be considered as feasible what ever their value is.

        Parameters
        ----------

        X : np.array
            A two dimensional matrix where each row is a point to evaluate and each column a variable.

        return_as_dictionary : bool
            If this is true than only one object, a dictionary, is returned. This contains all the results
            that are defined by return_values_of. Otherwise, by default a tuple as defined is returned.

        return_values_of : list of strings
            You can provide a list of strings which defines the values that are returned. By default it is set to
            "auto" which means depending on the problem the function values or additional the constraint violation (if
            the problem has constraints) are returned. Otherwise, you can provide a list of values to be returned.

            Allowed is ["F", "CV", "G", "dF", "dG", "dCV", "feasible"] where the d stands for
            derivative and h stands for hessian matrix.


        Returns
        -------

            A dictionary, if return_as_dictionary enabled, or a list of values as defined in return_values_of.

        """

        # call the callback of the problem
        if self.callback is not None:
            self.callback(X)

        # make the array at least 2-d - even if only one row should be evaluated
        only_single_value = len(np.shape(X)) == 1
        X = np.atleast_2d(X)

        # check the dimensionality of the problem and the given input
        if X.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' % (X.shape[1], self.n_var))

        # automatic return the function values and CV if it has constraints if not defined otherwise
        if type(return_values_of) == str and return_values_of == "auto":
            return_values_of = ["F"]
            if self.n_constr > 0:
                return_values_of.append("CV")

        # all values that are set in the evaluation function
        values_not_set = [val for val in return_values_of]

        # have a look if gradients are not set and try to use autograd and calculate grading if implemented using it
        gradients_not_set = [val for val in values_not_set if val.startswith("d")]

        # whether gradient calculation is necessary or not
        calc_gradient = (len(gradients_not_set) > 0)

        # set in the dictionary if the output should be calculated - can be used for the gradient
        out = {}
        for val in return_values_of:
            out[val] = None

        # calculate the output array - either elementwise or not. also consider the gradient
        # NOTE: pass return_values_of to evaluation function to avoid unnecessary computation
        if self.elementwise_evaluation:
            out = self._evaluate_elementwise(X, calc_gradient, out, *args, return_values_of=return_values_of, **kwargs)
        else:
            out = self._evaluate_batch(X, calc_gradient, out, *args, return_values_of=return_values_of, **kwargs)

            calc_gradient_of = [key for key, val in out.items()
                                if "d" + key in return_values_of and
                                out.get("d" + key) is None and
                                (type(val) == autograd.numpy.numpy_boxes.ArrayBox)]

            if len(calc_gradient_of) > 0:
                deriv = self._calc_gradient(out, calc_gradient_of)
                out = {**out, **deriv}

        # convert back to conventional numpy arrays - no array box as return type
        for key in out.keys():
            if type(out[key]) == autograd.numpy.numpy_boxes.ArrayBox:
                out[key] = out[key]._value

        # if constraint violation should be returned as well
        if self.n_constr == 0:
            CV = np.zeros([X.shape[0], 1])
        else:
            CV = Problem.calc_constraint_violation(out["G"])

        if "CV" in return_values_of:
            out["CV"] = CV

        # if an additional boolean flag for feasibility should be returned
        if "feasible" in return_values_of:
            out["feasible"] = (CV <= 0)

        # if asked for a value but not set in the evaluation set to None
        for val in return_values_of:
            if val not in out:
                out[val] = None

        # remove the first dimension of the output - in case input was a 1d- vector
        if only_single_value:
            for key in out.keys():
                if out[key] is not None:
                    out[key] = out[key][0, :]

        if return_as_dictionary:
            return out
        else:

            # if just a single value do not return a tuple
            if len(return_values_of) == 1:
                return out[return_values_of[0]]
            else:
                return tuple([out[val] for val in return_values_of])

    def _evaluate_batch(self, X, calc_gradient, out, *args, **kwargs):
        # NOTE: to use self-calculated dF (gradient) rather than autograd.numpy, which is not supported by Pymoo
        self._evaluate(X, out, *args, calc_gradient=calc_gradient, **kwargs)
        at_least_2d_array(out)
        return out

    def _evaluate_elementwise(self, X, calc_gradient, out, *args, **kwargs):
        # NOTE: to use self-calculated dF (gradient) rather than autograd.numpy, which is not supported by Pymoo
        ret = []
        
        popa=[]
        for i in range(X.shape[0]):
            
            _out = {}
            id=self._evaluate(X[i], _out, *args, **kwargs)
            ret.append(_out)
            popa.append(id)
        self.archive.append(popa)
            
        # stack all the single outputs together
        for key in ret[0].keys():
            out[key] = row_stack([ret[i][key] for i in range(len(ret))])
        #print(out)
        return out

    def _evaluate(self, x, out, *args, **kwargs):
        y1=-self.acc_surr.predict(x.reshape((-1,23)))[0]
        y2=self.lat_surr.predict(x.reshape((-1,23)))[0]
        out["F"]=np.array([y1,y2])
        return id


    def __str__(self):
        return '========== Problem Definition ==========\n' + super().__str__()