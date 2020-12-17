import os

import torch
from torch import nn


class ResidualGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout_p):
        super(ResidualGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.grus = nn.ModuleList([nn.GRU(2 * hidden_size, hidden_size,
                                          batch_first=True) for _ in range(self.num_layers)])
        self.output_layer = nn.Linear(self.hidden_size, output_size)

    def init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, x, hidden):
        x = nn.SELU()(self.input_layer(x))
        x_prev = x
        for i in range(len(self.grus)):
            gru_input = torch.cat((x_prev, x), dim=-1)
            x_prev = x
            x, hidden = self.grus[i](gru_input, hidden)
        x = self.output_layer(x)
        return x

    def save(self, archive_dir):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        torch.save({
            "state_dict": self.state_dict(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout_p": self.dropout_p
        }, os.path.join(archive_dir, "model.pt"))

    @staticmethod
    def load(archive_dir, device):
        print("Loading model from {}".format(archive_dir))
        checkpoint = torch.load(os.path.join(archive_dir, "model.pt"), map_location=device)
        model = ResidualGRU(checkpoint["input_size"], checkpoint["output_size"],
                             checkpoint["hidden_size"], checkpoint["num_layers"],
                             checkpoint["dropout_p"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        return model
