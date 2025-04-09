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
        self.states = np.random.choice([-1, 1], size=self.N)
        self.overlaps = np.zeros(self.P,dtype=float)

        self.generate_balanced_patterns()
        self.compute_overlap()

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

    def compute_overlap(self) -> list:
        ##complexity N*P
        for i in range(self.patterns.shape[0]):
            overlap = 0
            for j in range(len(self.states)):
                overlap += self.patterns[i][j] * self.states[j]
            self.overlaps[i] = (1 / len(self.patterns)) * overlap
        
        return self.overlaps

    def compute_next_state(self) -> list:
        """
        Ex.02
        The complexity of this is O(N^2 + N^2*P) --> N^2 to calculate the next state + N^2*P to calculate the weights matrix
        """
        curr_state = self.states.copy()
        curr_weights = self.get_weight_matrix()
        h = np.zeros(self.N, dtype=float)

        for i in range(self.N):
            h_i = np.dot(curr_weights[i], curr_state.T)
            new_state = np.sign(h_i)
            h[i] = new_state

        self.states = h
        return self.states

    def compute_next_state_fast(self) -> np.ndarray:
        """
        Ex0.3
        The complexity O(N*P + N*P) --> N*P to calculte the overlap array + N*P to calculate the next state
        """
        states = np.zeros(self.N, dtype=float)
        N = len(self.states)
        for i in range(len(self.states)):
            h = np.dot(np.array(self.overlaps), self.patterns[:, i])
            new_state = np.sign(h)
            states[i] = new_state
        self.states = states
        self.overlaps = self.compute_overlap()
        return self.states

    def compute_next_state_1(self):
        """
        Ex 0.3.
        Preferred to compute_next_state_0 when wanting to avoid storing the full NxN weight matrix, in which case storage and time complexity becomes O(N^2)
        """

        curr_state = self.states.copy()
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
        self.states = np.where(h == 0, 1, np.sign(h))

        return self.states


if __name__ == "__main__":
    N, P = 20, 2

    network = HopfieldNetwork(N, P)
    print(network.states)
    patterns = network.generate_balanced_patterns()
    print(patterns)
    weights = network.get_weight_matrix()
    print(weights)
    print(network.compute_next_state_1())
