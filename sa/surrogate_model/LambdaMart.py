import numpy as np
from sklearn.tree import DecisionTreeRegressor
from numba import jit

class LambdaMART:
    """Original paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf"""
    def __init__(self, num_trees=100, max_depth=3, learning_rate=1.0):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.trees = []
        self.gamma = np.zeros((num_trees, 2**(max_depth + 1) - 1))
        self.delta_ndcg_first_dict = {}
        
    def _idcg(self, rank):
        rank = np.sort(rank)[::-1]
        i = np.arange(1, len(rank)+1)
        return np.dot((2**rank - 1), 1/np.log2(i + 1))
        
    def _calculate_lambda(self, rank, F, qid):
        order = np.argsort(F)[::-1] + 1
        if qid in self.delta_ndcg_first_dict:
            delta_ndcg_first = self.delta_ndcg_first_dict[qid]
        else:
            idcg = self._idcg(rank)
            if idcg != 0:
                delta_ndcg_first = (2**rank - 1) / idcg
            else:
                delta_ndcg_first = np.zeros(len(rank))
            self.delta_ndcg_first_dict[qid] = delta_ndcg_first
        delta_s_matrix = np.reshape(F, (-1, 1)) - np.reshape(F, (1, -1))
        delta_ndcg_matrix = np.zeros((len(rank), len(rank)))
        log_order = 1 / np.log2(1 + order)
        for i in range(len(rank)):
            for j in range(i+1, len(rank)):
                if rank[i] != rank[j]:
                    log_order_swap = np.copy(log_order)
                    log_order_swap[i], log_order_swap[j] = log_order_swap[j], log_order_swap[i]
                    if rank[i] > rank[j]:
                        delta_ndcg_matrix[i, j] = np.dot(delta_ndcg_first, log_order_swap - log_order)
                    else:
                        delta_ndcg_matrix[j, i] = np.dot(delta_ndcg_first, log_order_swap - log_order)
        rho_matrix = 1 / (1 + np.exp(delta_s_matrix))
        abs_delta_ndcg_matrix = np.abs(delta_ndcg_matrix)
        omega_matrix = abs_delta_ndcg_matrix * rho_matrix
        lambda_matrix = -omega_matrix
        omega_matrix *= (1 - rho_matrix)
        lambda_matrix -= lambda_matrix.T
        omega_matrix += omega_matrix.T
        return np.sum(lambda_matrix, axis=0), np.sum(omega_matrix, axis=0)
    
    def fit(self, X, rank, qid):
        F = np.zeros(np.shape(X)[0])
        eps = 0.000001
        for k in range(self.num_trees):
            lambda_arr = np.array([])
            omega_arr = np.array([])
            for unique_qid in np.unique(qid):
                qid_lambda, qid_omega = self._calculate_lambda(rank[qid == unique_qid], 
                                                               F[qid == unique_qid], unique_qid)
                lambda_arr = np.append(lambda_arr, qid_lambda)
                omega_arr = np.append(omega_arr, qid_omega)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, lambda_arr)
            self.trees.append(tree)
            leaves = tree.apply(X)
            for leaf in np.unique(leaves):
                leaf_idx = (leaves == leaf)
                self.gamma[k, leaf] = np.sum(lambda_arr[leaf_idx]) / (np.sum(omega_arr[leaf_idx]) + eps)
                F += self.lr * leaf_idx * self.gamma[k, leaf]

    def predict(self, X):
        F = np.zeros(np.shape(X)[0])
        for k in range(len(self.trees)):
            leaves = self.trees[k].apply(X)
            for leaf in np.unique(leaves):
                leaf_idx = (leaves == leaf)
                F += self.lr * leaf_idx * self.gamma[k, leaf]
        return F