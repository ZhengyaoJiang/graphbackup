from src.rlpyt.rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from src.rlpyt.rlpyt.models.dqn.atari_catdqn_model import AtariCatDqnModel
from src.rlpyt.rlpyt.agents.dqn.atari.mixin import AtariMixin


class AtariCatDqnAgent(AtariMixin, CatDqnAgent):

    def __init__(self, ModelCls=AtariCatDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
