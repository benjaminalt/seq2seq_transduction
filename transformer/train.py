import argparse
import os
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from data import load_dataset
from transformer.model import TransformerModel
from utils import time_since, plot_loss_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset_path, batch_size, num_epochs, learning_rate):
    normalized_carrier, normalized_params, normalized_modulated, _, _ = load_dataset(dataset_path)
    dataset = TensorDataset(torch.from_numpy(normalized_carrier).float(),
                            torch.from_numpy(normalized_params).float(),
                            torch.from_numpy(normalized_modulated).float())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                             pin_memory=True, drop_last=True)
    model = TransformerModel(input_dim=normalized_carrier.shape[-1] + normalized_params.shape[-1], output_dim=normalized_modulated.shape[-1],
                             d_model=256, nhead=4).to(device)
    output_dir = os.path.join("output", "transformer")
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
            batch_loss = train_step(input_seq_batch, target_seq_batch, model, optimizer, criterion)
            total_loss += batch_loss

        avg_train_loss = total_loss / len(data_loader)
        print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, avg_train_loss))
        loss_history.append(avg_train_loss)

        if epoch % 10 == 0:
            model.save(os.path.join(checkpoint_dir, str(epoch)))

    return loss_history


def train_step(input_tensor, target_tensor, model, optimizer, criterion):
    batch_size, seq_len, input_size = input_tensor.size()

    optimizer.zero_grad()

    # Model expects (seq_len, batch_size, input_size)
    input_tensor = input_tensor.reshape((seq_len, batch_size, -1))
    target_tensor = target_tensor.reshape((seq_len, batch_size, -1))

    memory = model.encoder_forward(input_tensor)
    output = model.decoder_forward(target_tensor, memory)   # Full teacher forcing --> do I have to right shift target_tensor here?

    loss = criterion(output, target_tensor)

    loss.backward()
    optimizer.step()

    return loss.item()


def main(args):
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 100

    train(args.dataset_path, batch_size, num_epochs, learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
