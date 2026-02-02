import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm as sparse_expm
from scipy.linalg import expm as dense_expm
from scipy.sparse.csgraph import connected_components

class Solver:
    def solve(self, problem: dict[str, sparse.spmatrix], **kwargs) -> sparse.spmatrix:
        A = problem["matrix"]
        
        # Ensure CSC format
        if not isinstance(A, sparse.csc_matrix):
            A = A.tocsc()
        
        N = A.shape[0]
        
        # Optimization 1: Use dense solver for small matrices
        if N < 600:
            return sparse.csc_matrix(dense_expm(A.toarray()))
        
        # Optimization 2: Decompose into connected components
        n_comp, labels = connected_components(A, connection='weak')
        
        # Special case: Diagonal matrix
        if n_comp == N:
            return sparse.diags(np.exp(A.diagonal()), format='csc')
        
        if n_comp > 1:
            order = np.argsort(labels)
            A_perm = A[order, :][:, order]
            sorted_labels = labels[order]
            unique_labels, counts = np.unique(sorted_labels, return_counts=True)
            
            res_blocks = []
            current_idx = 0
            
            for count in counts:
                end_idx = current_idx + count
                block = A_perm[current_idx:end_idx, current_idx:end_idx]
                
                if count < 600:
                    blk_res = sparse.csc_matrix(dense_expm(block.toarray()))
                else:
                    blk_res = sparse_expm(block)
                
                res_blocks.append(blk_res)
                current_idx = end_idx
            
            res_perm = sparse.block_diag(res_blocks, format='csc')
            
            inv_order = np.empty_like(order)
            inv_order[order] = np.arange(N)
            
            return res_perm[inv_order, :][:, inv_order]
        
        return sparse_expm(A)