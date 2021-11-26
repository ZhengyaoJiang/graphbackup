from gym import ObservationWrapper

class DiscreteObservationWrapper(ObservationWrapper):
    def __init__(self, env, min, max, nb_bins):
        super(DiscreteObservationWrapper, self).__init__(env)
