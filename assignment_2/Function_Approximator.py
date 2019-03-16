import numpy as np
import Tiling

#abstract class of function approximator
class function_approximator:

    def __init__(self,tilings):
        self.tilings = tilings

    def estimated_value(self,state):
        pass

    def get_gradient(self,status):
        pass


class linear_function_approximator(function_approximator):

    def __init__(self,tilings,lb,ub,seed):
        function_approximator.__init__(self,tilings)
        self.lb = lb
        self.ub = ub
        self.seed = seed
        self.weights = self._initialize_weights()


    def _initialize_weights(self):
        np.random.seed(self.seed)
        state_sizes = [tuple(len(split)+1 for split in self.tilings[0])]
        return np.random.uniform(self.lb,self.ub,(len(self.tilings),)+state_sizes[0])

    def _get_features(self,state):
        encoded_state = Tiling.tile_encode(state, self.tilings, flatten=False)
        features = np.zeros_like(self.weights)
        for idx_tile, state_idx in enumerate(encoded_state):
            features[idx_tile][state_idx] = 1
        assert(np.sum(features) == len(self.tilings) )
        return features

    def estimated_value(self,state):
        features = self._get_features(state)
        return np.sum(np.multiply(features, self.weights))

    def get_gradient(self,state):
        return self._get_features(state)

    def update_weights(self,delta_weights):
        self.weights += delta_weights

    def reset_weights(self):
        self.weights*=0

