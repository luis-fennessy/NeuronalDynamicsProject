# ----------------------- Imports. ----------------------------

import numpy as np

# ----------------------- Ex 0. ----------------------------

# learning a bit about the intuition of Hopfield Networks via:
# https://www.youtube.com/watch?v=1WPJdAW-sFo

class HopfieldNetwork:
    """
    Class from which we derive the methods required for creating, inferencing and training a Hopfield Network
    """

    def __init__(self, N, P):
        self.N = N  # Number of neurons
        self.P = P  # Number of patterns
        self.patterns = np.zeros((self.P, self.N), dtype=int)
        self.state = np.random.choice([-1, 1], size=self.N)

    def generate_balanced_patterns(self):
        """
            Ex 0.1.
        """
        
        # as per Ex.0 "each pattern containing an equal number of +1s and -1s"
        assert self.N % 2 == 0
                
        for mu in range(self.P):
            # mu represents the index of the pattern amongst P patterns
            pattern = np.array([1] * (self.N // 2) + [-1] * (self.N // 2))
            np.random.shuffle(pattern)
            self.patterns[mu] = pattern
        
        return self.patterns

    
    def get_weight_matrix(self):
        """
            As per Eqn 1 under Ex 0.
        """

        W = np.zeros((self.N, self.N))
        for mu in range(self.P):
            W += np.outer(self.patterns[mu], self.patterns[mu])
        W /= self.N

        # in order for outcome of Ex0.2 == Ex0.3, we need diagonal weights = 0
        np.fill_diagonal(W, 0)
        
        return W


    def compute_next_state_0(self):
        """
            Ex 0.2.
        """

        curr_state = self.state.copy()
        curr_weights = self.get_weight_matrix()
        h = np.zeros(self.N, dtype=float)
        
        for i in range(self.N):
            for j in range(self.N):
                h[i] += curr_weights[i, j] * self.state[j]
            
        # can't have state as 0
        self.state = np.where(h == 0, 1, np.sign(h))

        return self.state
        

    def compute_next_state_1(self):
        """
            Ex 0.3.
            Preferred to compute_next_state_0 when wanting to avoid storing the full NxN weight matrix, in which case storage and time complexity becomes O(N^2)
        """

        curr_state = self.state.copy()
        m = np.zeros(self.P, dtype=float)
        h = np.zeros(self.N, dtype=float)
        
        for mu in range(self.P):
            sum = 0
            for i in range(self.N):
                sum += self.patterns[mu, i] * curr_state[i]
            m[mu] = sum / self.N

        for i in range(self.N):
            for mu in range(self.P):
                h[i] += m[mu] * self.patterns[mu, i]
            
        # can't have state as 0
        self.state = np.where(h == 0, 1, np.sign(h))

        return self.state


if __name__ == "__main__":
    N, P = 4, 2

    network = HopfieldNetwork(N, P)
    patterns = network.generate_balanced_patterns()
    # print(patterns)
    weights = network.get_weight_matrix()
    # print(weights)
    print(network.state)
    print(network.compute_next_state_1())


