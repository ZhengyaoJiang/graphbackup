import torch
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch.nn import functional as F

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MinigridNet(TorchModelV2, nn.Module):
    """Constructs the Student Policy which takes an observation and a goal and produces an action."""

    def __init__(self, obs_space, num_actions, num_outputs, model_config, name, num_input_frames=1, use_lstm=False,
                 num_lstm_layers=1, disable_embedding=False, no_generator=False):
        nn.Module.__init__(self)
        TorchModelV2.__init__(self, obs_space, num_actions, num_outputs,
                              model_config, name)
        state_embedding_dim = num_outputs
        self.observation_shape = obs_space.original_space.spaces["image"].shape
        self.num_actions = num_actions.n
        self.state_embedding_dim = state_embedding_dim
        self.use_lstm = use_lstm
        self.num_lstm_layers = num_lstm_layers

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim) * num_input_frames
        self.disable_embedding = disable_embedding
        self.no_generatro = no_generator

        if disable_embedding:
            print("not_using_embedding")
            self.num_channels = (3 + 1 + 1 + 1 + 1) * num_input_frames

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0] * self.observation_shape[1] + 1,
                                            self.agent_loc_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.obj_dim + self.col_dim, self.state_embedding_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.state_embedding_dim, self.state_embedding_dim)),
            nn.ReLU(),
        )

        if use_lstm:
            self.core = nn.LSTM(self.state_embedding_dim, self.state_embedding_dim, self.num_lstm_layers)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.baseline = init_(nn.Linear(self.state_embedding_dim, 1))

    def initial_state(self, batch_size):
        """Initializes LSTM."""
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:, :, :, id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:, :, :, id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:, :, :, id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape + (-1,))
        else:
            return embed(x)

    def agent_loc(self, frames):
        """Returns the location of an agent from an observation."""
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:, :, :, 0]
        agent_location = (agent_location == 10).nonzero()  # select object id
        agent_location = agent_location[:, 2]
        agent_location = agent_location.view(T, B, 1)
        return agent_location

    def forward(self, input_dict, state, seq_lens):
        """Main Function, takes an observation and a goal and returns and action."""

        # -- [unroll_length x batch_size x height x width x channels]
        x = input_dict["obs"]["image"]
        B, h, w, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]

        # Creating goal_channel
        carried_col = input_dict["obs"]["carried_col"]
        carried_obj = input_dict["obs"]["carried_obj"]

        if self.disable_embedding:
            x = x.float()
            carried_obj = carried_obj.float()
            carried_col = carried_col.float()
        else:
            x = x.long()
            carried_obj = carried_obj.long()
            carried_col = carried_col.long()
            # -- [B x H x W x K]
            x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2),
                           ], dim=3)
            carried_obj_emb = self._select(self.embed_object, carried_obj)
            carried_col_emb = self._select(self.embed_color, carried_col)

        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(B, -1)
        carried_obj_emb = carried_obj_emb.view(B, -1)
        carried_col_emb = carried_col_emb.view(B, -1)
        union = torch.cat([x, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)

        core_output = core_input
        self.lastest_core = core_output
        return core_output, []

    def value_function(self):
        return torch.squeeze(self.baseline(self.lastest_core))


