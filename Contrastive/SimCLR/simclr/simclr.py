import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SimCLR_Transformer(nn.Module):

    def __init__(self, projection_dim, n_channel, n_length=240, aug_mode=None):
        super(SimCLR_Transformer, self).__init__()

        """Build your own encoder, replace it by untrained Transformer"""
        # input data shape: [128, 2,240]
        if aug_mode == 'channel_wise':
            n_channel, n_length = 1, n_length
        else:
            n_channel, n_length = n_channel, n_length
        d_model = n_length
        n_head = 2
        n_hid = 512
        encoder_layers = TransformerEncoderLayer(d_model, n_head, n_hid, dropout=0.1, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layers, num_layers=2)
        self.n_features = n_channel*n_length

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, int(self.n_features/2), bias=True),
            nn.ReLU(),
            nn.Linear(int(self.n_features/2), projection_dim, bias=True),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        h_i = h_i.flatten(start_dim=1, end_dim=-1)
        h_j = h_j.flatten(start_dim=1, end_dim=-1)


        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j