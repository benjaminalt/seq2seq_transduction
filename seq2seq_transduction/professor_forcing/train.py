import argparse
import os
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from seq2seq_transduction.data import load_dataset
from seq2seq_transduction.professor_forcing import EncoderDecoder
from seq2seq_transduction.utils import time_since, plot_loss_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def need_update(tf_scores, ar_scores):
    """
    Discriminator accuracy < 0.75 --> don't backpropagate to generator
    Discriminator accuracy > 0.99 --> don't train discriminator
    Discriminator guess is calculated as x > 0.5
    :param tf_scores: Teacher Forcing scores [batch_size, 1]
    :param ar_scores: Autoregressive scores  [batch_size, 1]
    :return:
    """
    correct = float((tf_scores.view(-1) > 0.5).sum() + (ar_scores.view(-1) < 0.5).sum())
    d_accuracy = correct / (tf_scores.size(0) * 2)
    if d_accuracy < 0.75:
        return False, True
    elif d_accuracy > 0.99:
        return True, False
    else:
        return True, True


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
    model = EncoderDecoder(encoder_input_size, decoder_output_size, hidden_size,
                         num_layers, dropout_p=0.1, seq_len=seq_len, attention=False).to(device)
    output_dir = os.path.join("output", "professor_forcing")
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

    g_optim = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=learning_rate)
    d_optim = torch.optim.Adam(list(model.discriminator.parameters()), lr=learning_rate)

    model.train()

    for epoch in range(1, n_epochs + 1):
        total_g_loss = 0
        total_d_loss = 0
        for carrier_sig, params, modulated_sig in data_loader:
            param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
            input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
            target_seq_batch = modulated_sig.to(device)
            g_loss, d_loss = train_step(input_seq_batch, target_seq_batch, model, g_optim, d_optim)
            total_g_loss += g_loss
            total_d_loss += d_loss

        avg_g_loss = total_g_loss / len(data_loader)
        avg_d_loss = total_d_loss / len(data_loader)
        print('%s (%d %d%%) Generator: %.4f, discriminator: %.4f' % (time_since(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, avg_g_loss, avg_d_loss))
        loss_history.append((avg_g_loss, avg_d_loss))

        if epoch % 10 == 0:
            model.save(os.path.join(checkpoint_dir, str(epoch)))

    return loss_history


def train_step(input_tensor, target_tensor, model, g_optim, d_optim):
    batch_size, seq_len, _ = input_tensor.size()

    g_optim.zero_grad()
    d_optim.zero_grad()

    encoder_hidden = model.encoder.init_hidden(batch_size, device)

    encoder_outputs, encoder_hidden = model.encoder(input_tensor, encoder_hidden)

    decoder_input = torch.zeros(batch_size, 1, model.decoder.output_size, device=device)
    decoder_hidden = encoder_hidden

    # Teacher forcing
    tf_outputs = None
    tf_hiddens = None
    for di in range(seq_len):
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
        if di == 0:
            tf_outputs = decoder_output
            tf_hiddens = decoder_hidden
        else:
            tf_outputs = torch.cat((tf_outputs, decoder_output[:,-1,:].unsqueeze(1)), dim=1)
            tf_hiddens = torch.cat((tf_hiddens, decoder_hidden), dim=0)
        decoder_input = target_tensor[:, di, :].unsqueeze(1) # Teacher forcing

    # Autoregressive
    ar_outputs = None
    ar_hiddens = None
    for di in range(seq_len):
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
        if ar_outputs is None:
            ar_outputs = decoder_output
            ar_hiddens = decoder_hidden
        else:
            ar_outputs = torch.cat((ar_outputs, decoder_output[:,-1,:].unsqueeze(1)), dim=1)
            ar_hiddens = torch.cat((ar_hiddens, decoder_hidden), dim=0)
        decoder_input = decoder_output.detach()  # detach from history as input

    tf_scores = model.discriminator(tf_hiddens)
    ar_scores = model.discriminator(ar_hiddens)

    tf_loss = nn.MSELoss(reduction="none")(tf_outputs, target_tensor).mean(dim=-1)
    ar_loss = (- torch.log(ar_scores) - torch.log(1 - tf_scores))
    generator_loss = (tf_loss + ar_loss).sum()
    discriminator_loss = (- torch.log(tf_scores) - torch.log(1 - ar_scores)).sum()

    update_g, update_d = need_update(tf_scores, ar_scores)
    if not update_g:
        tf_hiddens.detach()
        ar_hiddens.detach()
    discriminator_loss.backward(retain_graph=True)
    generator_loss.backward()

    g_optim.step()
    d_optim.step()

    return generator_loss.item(), discriminator_loss.item()


def main(args):
    batch_size = 128
    hidden_size = 256
    learning_rate = 0.0001
    num_epochs = 100
    num_layers = 4

    train(args.dataset_path, batch_size, hidden_size, num_layers, num_epochs, learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
