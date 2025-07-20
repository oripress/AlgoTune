import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        S = problem["num_states"]
        A = problem["num_actions"]
        g = problem["discount"]
        # load arrays
        P = np.asarray(problem["transitions"])
        R_full = np.asarray(problem["rewards"])
        # expected immediate rewards
        R_sa = (P * R_full).sum(axis=2)
        # initial policy: greedy on immediate reward
        policy = R_sa.argmax(axis=1)
        # prepare indices and identity matrix
        idx = np.arange(S)
        I = np.eye(S)
        # policy iteration
        while True:
            # policy evaluation: solve (I - g*P_pi) V = R_pi
            P_pi = P[idx, policy, :]
            R_pi = R_sa[idx, policy]
            V = np.linalg.solve(I - g * P_pi, R_pi)
            # policy improvement: Q = R_sa + g * P·V
            Q = R_sa + g * (P.dot(V))
            new_policy = Q.argmax(axis=1)
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy
        return {"value_function": V.tolist(), "policy": policy.tolist()}