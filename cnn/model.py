import os

import torch
from torch import nn


class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout_p):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.num_channels = 32

        self.cnn_1 = nn.Conv1d(input_size, self.num_channels, kernel_size=256)
        self.cnn_2 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=128)
        self.cnn_3 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=64)
        self.cnn_4 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=32)
        self.cnn_5 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=16)
        self.cnn_6 = nn.Conv1d(self.num_channels, self.num_channels, kernel_size=8)
        self.gru = nn.GRU(input_size=self.input_size + self.num_channels, hidden_size=self.hidden_size,
                          num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, orig_input):
        batch_size = orig_input.size(0)
        seq_len = orig_input.size(1)
        features = orig_input.reshape((batch_size, orig_input.size(2), seq_len))
        features = nn.SELU()(self.cnn_1(features))
        features = nn.Dropout(self.dropout_p)(features)
        features = nn.SELU()(self.cnn_2(features))
        features = nn.Dropout(self.dropout_p)(features)
        features = nn.SELU()(self.cnn_3(features))
        features = nn.Dropout(self.dropout_p)(features)
        features = nn.SELU()(self.cnn_4(features))
        features = nn.Dropout(self.dropout_p)(features)
        features = nn.SELU()(self.cnn_5(features))
        features = nn.Dropout(self.dropout_p)(features)
        features = nn.SELU()(self.cnn_6(features))
        features = features.reshape((batch_size, features.size(2), -1))
        init_hidden = torch.zeros(1, batch_size, self.hidden_size, device=features.device)
        feature_seq = torch.cat((features, torch.zeros(batch_size, seq_len - features.size(1), features.size(2),
                                                       device=orig_input.device)), dim=1)
        gru_input = torch.cat((orig_input, feature_seq), dim=-1)
        gru_output, _ = self.gru(gru_input, init_hidden)
        output = self.output_layer(gru_output)
        return output

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
        model = CNNModel(checkpoint["input_size"], checkpoint["output_size"], checkpoint["hidden_size"],
                         checkpoint["num_layers"], checkpoint["dropout_p"])
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        return model
