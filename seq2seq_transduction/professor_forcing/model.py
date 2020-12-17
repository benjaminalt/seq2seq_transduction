import os

import torch
from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_input_size, decoder_output_size, hidden_size,
                 num_layers, dropout_p, seq_len, attention):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderRNN(encoder_input_size, hidden_size, num_layers)
        self.decoder = AttnDecoderRNN(hidden_size, decoder_output_size, num_layers,
                                      dropout_p, seq_len, attention)
        self.discriminator = Discriminator(input_size=hidden_size)

    def save(self, archive_dir):
        print("Saving model to {}".format(archive_dir))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        torch.save({
            "state_dict": self.state_dict(),
            "encoder_input_size": self.encoder.input_size,
            "decoder_output_size": self.decoder.output_size,
            "hidden_size": self.encoder.hidden_size,
            "num_layers": self.encoder.num_layers,
            "dropout_p": self.decoder.dropout_p,
            "seq_len": self.decoder.max_length,
            "attention": self.decoder.attention
        }, os.path.join(archive_dir, "model.pt"))

    @staticmethod
    def load(archive_dir, device):
        print("Loading model from {}".format(archive_dir))
        checkpoint = torch.load(os.path.join(archive_dir, "model.pt"), map_location=device)
        model = Generator(checkpoint["encoder_input_size"], checkpoint["decoder_output_size"],
                          checkpoint["hidden_size"], checkpoint["num_layers"],
                          checkpoint["dropout_p"], checkpoint["seq_len"], checkpoint["attention"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.linear = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=self.num_layers)

    def forward(self, inp, hidden):
        gru_input = self.linear(inp)
        output, hidden = self.gru(gru_input, hidden)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers,
                 dropout_p=0.1, max_length=500, attention=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention = attention

        self.linear = nn.Linear(self.output_size, self.hidden_size)
        if self.attention:
            self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
            self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp, hidden, encoder_outputs):
        batch_size = inp.size(0)

        embedded = self.dropout(nn.SELU()(self.linear(inp)))

        if self.attention:
            attn_inputs = torch.cat((embedded, hidden[-1, :, :].view(batch_size, 1, hidden.size(-1))), -1)
            attn_weights = torch.nn.Softmax(dim=-1)(self.attn(attn_inputs))
            attn_applied = torch.bmm(attn_weights, encoder_outputs)
            output = torch.cat((embedded, attn_applied), -1)
            output = self.attn_combine(output)
            output = torch.nn.SELU()(output)
        else:
            output = embedded

        output, hidden = self.gru(output, hidden)

        output = self.out(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional_gru = nn.GRU(self.input_size, self.hidden_size, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_size, 2048),
            nn.SELU(),
            nn.Linear(2048, 2048),
            nn.SELU(),
            nn.Linear(2048, 2048),
            nn.SELU()
        )
        self.output = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, b):
        """
        :param b: The behavior sequence
        :return: Discriminator output D(b)
        """
        batch_size = b.size(1)
        init_hidden = torch.zeros(self.bidirectional_gru.num_layers * 2, batch_size, self.bidirectional_gru.hidden_size)
        gru_output, hidden = self.bidirectional_gru(b, init_hidden)
        mlp_output = self.mlp(hidden.view(hidden.size(1), hidden.size(0) * hidden.size(-1)))
        return self.output(mlp_output)
