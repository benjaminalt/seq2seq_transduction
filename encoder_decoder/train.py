import argparse
import os
import random
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from data import load_dataset
from encoder_decoder.model import Seq2SeqModel
from utils import time_since, plot_loss_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def scheduled_sampling(epoch_idx, n_epochs):
    """
    See arXiv:1506.03099.
    Decide whether or not to use teacher forcing with a probability decreasing during training.
    :return: True for teacher forcing, False otherwise
    """
    teacher_forcing_prob = 1 - epoch_idx / n_epochs
    return random.random() < teacher_forcing_prob


def train(dataset_path, batch_size, hidden_size, num_layers, num_epochs, learning_rate):
    carrier_train, params_train, modulated_train, _, _ = load_dataset(dataset_path, "train")
    carrier_validate, params_validate, modulated_validate, _, _ = load_dataset(dataset_path, "validate")
    seq_len = carrier_train.shape[1]
    train_dataset = TensorDataset(torch.from_numpy(carrier_train).float(),
                                  torch.from_numpy(params_train).float(),
                                  torch.from_numpy(modulated_train).float())
    validate_dataset = TensorDataset(torch.from_numpy(carrier_validate).float(),
                                     torch.from_numpy(params_validate).float(),
                                     torch.from_numpy(modulated_validate).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                              pin_memory=True, drop_last=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                 pin_memory=True, drop_last=True)
    encoder_input_size = carrier_train.shape[-1] + params_train.shape[-1]
    decoder_output_size = modulated_train.shape[-1]
    model = Seq2SeqModel(encoder_input_size, decoder_output_size, hidden_size,
                         num_layers, dropout_p=0.1, seq_len=seq_len, attention=False).to(device)
    output_dir = os.path.join("output", "encoder_decoder")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    loss_history = train_loop(model, train_loader, validate_loader, num_epochs, learning_rate, checkpoint_dir)
    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    model.save(os.path.join(output_dir, timestamp))
    plot_loss_history(loss_history, os.path.join(output_dir, "{}_loss.png".format(timestamp)), labels=["Train", "Validate"])


def train_loop(model, train_loader, validate_loader, n_epochs, learning_rate, checkpoint_dir):
    start = time.time()
    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):

        total_loss = 0
        model.train()
        for carrier_sig, params, modulated_sig in train_loader:
            param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
            input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
            target_seq_batch = modulated_sig.to(device)
            use_teacher_forcing = scheduled_sampling(epoch, n_epochs)
            total_loss += train_step(input_seq_batch, target_seq_batch, model, optimizer, criterion, use_teacher_forcing)
        avg_train_loss = total_loss / len(train_loader)

        total_loss = 0
        model.eval()
        with torch.no_grad():
            for carrier_sig, params, modulated_sig in validate_loader:
                param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
                input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
                target_seq_batch = modulated_sig.to(device)
                total_loss += train_step(input_seq_batch, target_seq_batch, model, optimizer, criterion, False, True)
        avg_validate_loss = total_loss / len(validate_loader)
        print('%s (%d %d%%) Train: %.4f Validate: %.4f' % (time_since(start, epoch / n_epochs),
                                                           epoch, epoch / n_epochs * 100, avg_train_loss,
                                                           avg_validate_loss))
        loss_history.append((avg_train_loss, avg_validate_loss))

        if epoch % 10 == 0:
            model.save(os.path.join(checkpoint_dir, str(epoch)))

    return loss_history


def train_step(input_tensor, target_tensor, model, optimizer, criterion, use_teacher_forcing, validate=False):
    batch_size, seq_len, _ = input_tensor.size()

    optimizer.zero_grad()

    encoder_hidden = model.encoder.init_hidden(batch_size, device)
    encoder_outputs = torch.zeros(batch_size, seq_len, model.encoder.hidden_size, device=device)

    for ei in range(seq_len):
        encoder_output, encoder_hidden = model.encoder(input_tensor[:, ei, :].unsqueeze(1), encoder_hidden)
        encoder_outputs[:, ei] = encoder_output[:, 0]

    decoder_input = torch.zeros(batch_size, 1, model.decoder.output_size, device=device)
    decoder_hidden = encoder_hidden

    decoder_outputs = None

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(seq_len):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            if decoder_outputs is None:
                decoder_outputs = decoder_output
            else:
                decoder_outputs = torch.cat((decoder_outputs, decoder_output[:, -1, :].unsqueeze(1)), dim=1)
            decoder_input = target_tensor[:, di, :].unsqueeze(1)  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(seq_len):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            if decoder_outputs is None:
                decoder_outputs = decoder_output
            else:
                decoder_outputs = torch.cat((decoder_outputs, decoder_output[:, -1, :].unsqueeze(1)), dim=1)
            decoder_input = decoder_output.detach()  # detach from history as input

    loss = criterion(decoder_outputs, target_tensor)

    if not validate:
        loss.backward()
        optimizer.step()

    return loss.item()


def main(args):
    batch_size = 180
    hidden_size = 256
    learning_rate = 0.0001
    num_epochs = 150
    num_layers = 4

    train(args.dataset_path, batch_size, hidden_size, num_layers, num_epochs, learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
