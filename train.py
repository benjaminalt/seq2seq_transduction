import os
import random
import time
from datetime import datetime

import torch
from torch import nn
import joblib

from data import generate_sins, amplitude_modulate, normalize
from model import Seq2SeqModel
from utils import time_since, plot_loss_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = os.path.abspath(os.path.dirname(__file__))

SEQ_LEN = 100
BATCH_SIZE = 128


def save(model, signal_scaler, param_scaler, archive_dir):
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    torch.save({
        "model": model.state_dict()
    }, os.path.join(archive_dir, "model.pt"))
    joblib.dump(signal_scaler, os.path.join(archive_dir, "signal_scaler.pkl"))
    joblib.dump(param_scaler, os.path.join(archive_dir, "param_scaler.pkl"))


def train_iters(model, data_loader, n_epochs, learning_rate, signal_scaler, param_scaler, checkpoint_dir):
    start = time.time()
    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for carrier_sig, params, modulated_sig in data_loader:
            param_seq_batch = params.unsqueeze(1).repeat(1, SEQ_LEN, 1)
            input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
            target_seq_batch = modulated_sig.to(device)
            batch_loss = train(input_seq_batch, target_seq_batch, model, optimizer, criterion)
            total_loss += batch_loss

        avg_train_loss = total_loss / len(data_loader)
        print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, avg_train_loss))
        loss_history.append(avg_train_loss)

        if epoch % 10 == 0:
            save(model, signal_scaler, param_scaler, os.path.join(checkpoint_dir, str(epoch)))

    return loss_history


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, model, optimizer, criterion, seq_length=SEQ_LEN):
    encoder_hidden = model.encoder.init_hidden(BATCH_SIZE, device)

    optimizer.zero_grad()

    input_length = input_tensor.size(1)
    target_length = target_tensor.size(1)
    encoder_outputs = torch.zeros(BATCH_SIZE, seq_length, model.encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = model.encoder(input_tensor[:,ei,:].unsqueeze(1), encoder_hidden)
        encoder_outputs[:, ei] = encoder_output[:, 0]

    decoder_input = torch.zeros(BATCH_SIZE, 1, model.decoder.output_size, device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            teacher_signal = target_tensor[:,di].unsqueeze(1)
            loss += criterion(decoder_output, teacher_signal)
            decoder_input = teacher_signal # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:,di].unsqueeze(1))
            decoder_input = decoder_output.detach()  # detach from history as input

    loss.backward()
    optimizer.step()

    return loss.item() / target_length


def main():
    hidden_size = 256
    learning_rate = 0.0001
    num_epochs = 11
    num_data = 1000

    _, carrier_signals = generate_sins(num_data, SEQ_LEN)
    params, amplitudes = generate_sins(num_data, SEQ_LEN)
    modulated_signals = amplitude_modulate(carrier_signals, amplitudes)
    normalized_carrier, signal_scaler = normalize(carrier_signals)
    normalized_params, param_scaler = normalize(params)
    normalized_modulated, _ = normalize(modulated_signals, signal_scaler)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized_carrier).float(),
                                             torch.from_numpy(normalized_params).float(),
                                             torch.from_numpy(normalized_modulated).float())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                                              pin_memory=True, drop_last=True)

    encoder_input_size = carrier_signals.shape[-1] + params.shape[-1]
    decoder_output_size = modulated_signals.shape[-1]
    model = Seq2SeqModel(encoder_input_size, decoder_output_size, hidden_size, dropout_p=0.1, seq_len=SEQ_LEN).to(device)

    checkpoint_dir = os.path.join(script_dir, "output", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    loss_history = train_iters(model, data_loader, num_epochs, learning_rate, signal_scaler, param_scaler, checkpoint_dir)
    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    save(model, signal_scaler, param_scaler, os.path.join(script_dir, "output", timestamp))
    plot_loss_history(loss_history, os.path.join(script_dir, "output", "{}_loss.png".format(timestamp)))


if __name__ == '__main__':
    main()
