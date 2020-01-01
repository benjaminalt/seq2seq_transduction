import os

import torch
from torch import nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_decoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.input_dim = input_dim
        self.output_dim = output_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder_input_layer = nn.Linear(output_dim, d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.decoder_output_layer = nn.Linear(256, output_dim)

        self._reset_parameters()

    def encoder_forward(self, x):
        x = nn.SELU()(self.encoder_input_layer(x))
        return self.encoder(x)

    def decoder_forward(self, tgt, memory):
        tgt = nn.SELU()(self.decoder_input_layer(tgt))
        output = self.decoder(tgt, memory)
        return self.decoder_output_layer(output)

    def save(self, archive_dir):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward
        }, os.path.join(archive_dir, "model.pt"))

    @staticmethod
    def load(archive_dir, device):
        print("Loading model from {}".format(archive_dir))
        checkpoint = torch.load(os.path.join(archive_dir, "model.pt"), map_location=device)
        num_encoder_layers = checkpoint["num_encoder_layers"] if "num_encoder_layers" in checkpoint.keys() else 6
        num_decoder_layers = checkpoint["num_decoder_layers"] if "num_decoder_layers" in checkpoint.keys() else 6
        dim_feedforward = checkpoint["dim_feedforward"] if "dim_feedforward" in checkpoint.keys() else 2048

        model = TransformerModel(checkpoint["input_dim"], checkpoint["output_dim"],
                                 checkpoint["d_model"], checkpoint["nhead"],
                                 num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                 dim_feedforward=dim_feedforward)
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
