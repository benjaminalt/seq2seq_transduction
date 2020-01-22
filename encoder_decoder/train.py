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
    normalized_carrier, normalized_params, normalized_modulated, _, _ = load_dataset(dataset_path)
    seq_len = normalized_carrier.shape[1]
    dataset = TensorDataset(torch.from_numpy(normalized_carrier).float(),
                            torch.from_numpy(normalized_params).float(),
                            torch.from_numpy(normalized_modulated).float())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                             pin_memory=True, drop_last=True)
    encoder_input_size = normalized_carrier.shape[-1] + normalized_params.shape[-1]
    decoder_output_size = normalized_modulated.shape[-1]
    model = Seq2SeqModel(encoder_input_size, decoder_output_size, hidden_size,
                         num_layers, dropout_p=0.1, seq_len=seq_len, attention=False).to(device)
    output_dir = os.path.join("output", "encoder_decoder")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    loss_history = train_loop(model, data_loader, num_epochs, learning_rate, checkpoint_dir)
    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    model.save(os.path.join(output_dir, timestamp))
    plot_loss_history(loss_history, os.path.join(output_dir, "{}_loss.png".format(timestamp)))


def train_loop(model, data_loader, n_epochs, learning_rate, checkpoint_dir):
    start = time.time()
    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for carrier_sig, params, modulated_sig in data_loader:
            param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
            input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
            target_seq_batch = modulated_sig.to(device)
            use_teacher_forcing = scheduled_sampling(epoch, n_epochs)
            batch_loss = train_step(input_seq_batch, target_seq_batch, model, optimizer, criterion, use_teacher_forcing)
            total_loss += batch_loss

        avg_train_loss = total_loss / len(data_loader)
        print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, avg_train_loss))
        loss_history.append(avg_train_loss)

        if epoch % 10 == 0:
            model.save(os.path.join(checkpoint_dir, str(epoch)))

    return loss_history


def train_step(input_tensor, target_tensor, model, optimizer, criterion, use_teacher_forcing):
    batch_size, seq_len, _ = input_tensor.size()

    optimizer.zero_grad()

    encoder_hidden = model.encoder.init_hidden(batch_size, device)
    encoder_outputs = torch.zeros(batch_size, seq_len, model.encoder.hidden_size, device=device)

    for ei in range(seq_len):
        encoder_output, encoder_hidden = model.encoder(input_tensor[:,ei,:].unsqueeze(1), encoder_hidden)
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
                decoder_outputs = torch.cat((decoder_outputs, decoder_output[:,-1,:].unsqueeze(1)), dim=1)
            decoder_input = target_tensor[:, di, :].unsqueeze(1) # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(seq_len):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            if decoder_outputs is None:
                decoder_outputs = decoder_output
            else:
                decoder_outputs = torch.cat((decoder_outputs, decoder_output[:,-1,:].unsqueeze(1)), dim=1)
            decoder_input = decoder_output.detach()  # detach from history as input

    loss = criterion(decoder_outputs, target_tensor)

    loss.backward()
    optimizer.step()

    return loss.item()


def main(args):
    batch_size = 150
    hidden_size = 256
    learning_rate = 0.0001
    num_epochs = 200
    num_layers = 4

    train(args.dataset_path, batch_size, hidden_size, num_layers, num_epochs, learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
