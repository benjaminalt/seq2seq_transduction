import os

import torch
from torch import nn


class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers, dropout_p):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p

        self.num_channels = 32

        self.cnn_1 = nn.Conv1d(input_size, self.num_channels, kernel_size=128)
        self.cnn_2 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=64)
        self.cnn_3 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=32)
        self.cnn_4 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=16)
        self.cnn_5 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=8)
        self.gru = nn.GRU(self.num_channels, hidden_size=32, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.num_channels, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.reshape((batch_size, x.size(2), seq_len))
        x = nn.SELU()(self.cnn_1(x))
        x = nn.Dropout(self.dropout_p)(x)
        x = nn.SELU()(self.cnn_2(x))
        x = nn.Dropout(self.dropout_p)(x)
        x = nn.SELU()(self.cnn_3(x))
        x = nn.Dropout(self.dropout_p)(x)
        x = nn.SELU()(self.cnn_4(x))
        x = nn.Dropout(self.dropout_p)(x)
        x = nn.SELU()(self.cnn_5(x))
        x = x.reshape((batch_size, x.size(2), -1))
        init_hidden = torch.zeros(1, batch_size, 32, device=x.device)
        x = torch.cat((x, torch.zeros(batch_size, seq_len - x.size(1), x.size(2), device=x.device)), dim=1)
        x, _ = self.gru(x, init_hidden)
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
            "num_layers": self.num_layers,
            "dropout_p": self.dropout_p
        }, os.path.join(archive_dir, "model.pt"))

    @staticmethod
    def load(archive_dir, device):
        print("Loading model from {}".format(archive_dir))
        checkpoint = torch.load(os.path.join(archive_dir, "model.pt"), map_location=device)
        model = CNNModel(checkpoint["input_size"], checkpoint["output_size"],
                         checkpoint["hidden_size"], checkpoint["num_layers"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
