# ----------------------- Imports. ----------------------------

import numpy as np
from math import atan

# ----------------------- Ex 0. ----------------------------

# learning a bit about the intuition of Hopfield Networks via:
# https://www.youtube.com/watch?v=1WPJdAW-sFo


class HopfieldNetwork:
    """
    Class from which we derive the methods required for creating, inferencing and training a Hopfield Network
    """

    def __init__(self, N, P, K=None):
        self.N = N  # Number of neurons
        self.P = P  # Number of patterns
        self.patterns = np.zeros((self.P, self.N), dtype=int)
        self.states = np.random.choice([-1., 1.], size=self.N)
        self.overlaps = [0] * self.P
        self.dilution_mask = self.generate_dilution_mask(K)

        self.generate_balanced_patterns()
        self.compute_overlap()

    def generate_dilution_mask(self,K):
        if not K:
            # the default K is None, the network is not diluted
            return np.ones((self.N, self.N), dtype=int)
        
        C = np.zeros((self.N, self.N), dtype=int)

        for i in range(self.N):
            possible_indices = list(range(self.N))
            possible_indices.remove(i)  # exclude self-connection
            selected = np.random.choice(possible_indices, int(K), replace=False)
            C[i, selected] = 1

        return C
    
    def set_states(self,states):
        self.states = states

    def set_patterns(self,patterns):
        self.patterns = patterns

    def set_overlaps(self,overlaps):
        self.overlaps = overlaps

    def generate_balanced_patterns(self):
        """
        Ex 0.1.
        """

        # as per Ex.0 "each pattern containing an equal number of +1s and -1s"
        assert self.N % 2 == 0

        for mu in range(self.P):
            # mu represents the index of the pattern amongst P patterns
            pattern = np.array([1.] * (self.N // 2) + [-1.] * (self.N // 2))
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

    def compute_overlap(self, return_none=False, ex3=False) -> list:
        ##complexity N*P
        overlaps = [0] * self.P
        for i in range(self.P):
            overlap = 0
            for j in range(self.N):
                overlap += self.patterns[i][j] * self.states[j]
            overlaps[i] = (1 / len(self.states)) * overlap
            if ex3:
                # as per note in Ex3.1
                overlaps[i] *= 2
        
        self.overlaps = overlaps
        if not return_none:
            return self.overlaps

    def compute_next_state_ex3(self, beta, implement_refractory=False):
        """
        for exercise 3, calculate the firing probability from which we generate the next state
        """
        states = np.zeros(self.N, dtype=float)

        for i in range(self.N):
            # divide h_i by 2 as per Note 2 under Ex3.1
            h_i = np.dot(np.array(self.overlaps), self.patterns[:, i]) / 2
            Ph_i = 0.5 * (1 + atan(beta * h_i))

            # use random uniform dist to get state. If P(h_i) is high, random float is more likely less than it => state more likely = 1
            # refractory implemented below
            new_state = np.where((np.random.uniform(0, 1, 1) < Ph_i) & ((states[i] == 0) | (~implement_refractory)), 1, 0)
            states[i] = new_state

        self.states = states
        self.overlaps = self.compute_overlap(ex3=True)
        return self.states

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
    
    def compute_next_state_sparse(self) -> list:
        """
        Ex.2.2
        Using an adjusted weight matrix where there is 0 for weights between non-connnected neurons
        """
        curr_state = self.states.copy()
        curr_weights = self.get_weight_matrix()
        curr_weights = curr_weights * self.dilution_mask
        h = np.zeros(self.N, dtype=float)

        for i in range(self.N):
            h_i = np.dot(curr_weights[i], curr_state.T)
            new_state = np.sign(h_i)
            h[i] = new_state

        self.states = h
        self.overlaps = self.compute_overlap()
        
        return self.states

    def compute_next_state_fast(self):
        """
        Ex0.3
        The complexity O(N*P + N*P) --> N*P to calculte the overlap array + N*P to calculate the next state
        """
        states = np.zeros(self.N, dtype=float)

        for i in range(len(self.states)):
            h = np.dot(np.array(self.overlaps), self.patterns[:, i])
            new_state = np.where(h == 0, 1, np.sign(h))
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


# if __name__ == "__main__":
#     N, P = 30, 1

#     network = HopfieldNetwork(N, P)
#     print(network.states)
#     print(network.patterns)
#     weights = network.get_weight_matrix()
#     print("---------------------------------------")
#     print(network.compute_next_state_1())
#     print(network.compute_next_state_1())
#     print(network.compute_next_state_1())
#     print("----------------------------------")
#     print(network.patterns)
