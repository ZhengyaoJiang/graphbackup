from rainbow.memory import SegmentTree
from core.util import hashing
import numpy as np
import torch

class State2Index:
    def __init__(self, hashing_method):
        self.data = dict()
        self.hashing_method = hashing_method
        self.max = 0
        self.states = []

    def get_index(self, state):
        if isinstance(state, torch.Tensor):
            state = state.permute([1, 2, 0]).cpu().numpy()
        key = hashing(state, self.hashing_method)
        if key in self.data:
            return self.data[key]
        else:
            return None

    def get_indexes(self, states):
        return [self.get_index(state) for state in states]

    def get_states(self, indexs):
        return [self.states[i] for i in indexs]

    def append_state(self, state):
        key = hashing(state, self.hashing_method)
        if key not in self.data:
            self.data[key] = len(self.data)
            self.states.append(state)
            self.max += 1
            return False
        else:
            return True
