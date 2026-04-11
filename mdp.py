import numpy as np

class MDP:
    def __init__(self, P, R, gamma, mu):
        self.P = P          # (S, A, S)
        self.R = R          # (S, A)
        self.gamma = gamma
        self.mu = mu

        self.S = P.shape[0]
        self.A = P.shape[1]

    def P_pi(self, pi):
        return np.einsum('sa,sab->sb', pi, self.P)

    def R_pi(self, pi):
        return np.sum(pi * self.R, axis=1)

    def value_function(self, pi):
        P_pi = self.P_pi(pi)
        R_pi = self.R_pi(pi)
        I = np.eye(self.S)
        return np.linalg.solve(I - self.gamma * P_pi, R_pi)

    def Q_function(self, pi):
        v = self.value_function(pi)
        return self.R + self.gamma * np.einsum('sab,b->sa', self.P, v)

    def occupancy(self, pi):
        P_pi = self.P_pi(pi)
        I = np.eye(self.S)
        return self.mu @ np.linalg.inv(I - self.gamma * P_pi)

    def optimal_policy_value_iteration(self, max_iter=1000, tol=1e-6):
        """
        Compute optimal policy and value using value iteration.
        Returns: (optimal_policy, optimal_value_function, optimal_return)
        """
        V = np.zeros(self.S)
        
        for _ in range(max_iter):
            V_old = V.copy()
            
            # Compute Q-values for all state-action pairs
            Q = self.R + self.gamma * np.einsum('sab,b->sa', self.P, V)
            
            # Extract optimal values (max over actions)
            V = np.max(Q, axis=1)
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < tol:
                break
        
        # Compute optimal policy (greedy w.r.t. final V)
        Q = self.R + self.gamma * np.einsum('sab,b->sa', self.P, V)
        pi_optimal = np.zeros((self.S, self.A))
        for s in range(self.S):
            pi_optimal[s, np.argmax(Q[s])] = 1.0
        
        # Compute optimal return
        optimal_return = float(np.dot(self.mu, V))
        
        return pi_optimal, V, optimal_return
