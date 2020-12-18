import argparse
import os
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from seq2seq_transduction.transformer.model import Transformer
from seq2seq_transduction.data import load_dataset
from seq2seq_transduction.utils import time_since, plot_loss_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataset_path, batch_size, num_epochs, learning_rate):
    normalized_carrier_train, normalized_params_train, normalized_modulated_train, _, _ = load_dataset(dataset_path, kind="train")
    train_dataset = TensorDataset(torch.from_numpy(normalized_carrier_train).float(),
                            torch.from_numpy(normalized_params_train).float(),
                            torch.from_numpy(normalized_modulated_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                             pin_memory=True, drop_last=True)

    normalized_carrier_validate, normalized_params_validate, normalized_modulated_validate, _, _ = load_dataset(dataset_path, kind="validate")
    validate_dataset = TensorDataset(torch.from_numpy(normalized_carrier_validate).float(),
                            torch.from_numpy(normalized_params_validate).float(),
                            torch.from_numpy(normalized_modulated_validate).float())
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                             pin_memory=True, drop_last=True)

    p = {
        "enc_seq_len": 64,
        "dec_seq_len": 64,
        "input_size": 1 + 3,
        "output_sequence_length": 250,
        "dim_val": 32,
        "dim_attn": 32,
        "lr": learning_rate,
        "epochs": num_epochs,
        "n_heads": 16,
        "n_decoder_layers": 4,
        "n_encoder_layers": 4,
        "batch_size": batch_size
    }

    model = Transformer(p["dim_val"], p["dim_attn"], p["input_size"], p["dec_seq_len"], p["output_sequence_length"],
                        p["n_decoder_layers"], p["n_encoder_layers"], p["n_heads"]).to(device)
    output_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir, "output", "transformer")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    loss_history = train_loop(model, train_loader, validate_loader, num_epochs, learning_rate)
    timestamp = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    torch.save({"parameters": p, "state_dict": model.state_dict()}, os.path.join(output_dir, f"{timestamp}_model.pt"))
    plot_loss_history(loss_history, os.path.join(output_dir, "{}_loss.png".format(timestamp)), labels=["Train", "Validate"])


def train_loop(model, train_loader, val_loader, n_epochs, learning_rate):
    start = time.time()
    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        model.train()
        for carrier_sig, params, modulated_sig in tqdm(train_loader):
            param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
            input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
            target_seq_batch = modulated_sig.to(device)
            batch_loss = train_step(input_seq_batch, target_seq_batch, model, optimizer, criterion, validate=False)
            train_loss += batch_loss

        avg_train_loss = train_loss / len(train_loader)
        print('%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs),
                                     epoch, epoch / n_epochs * 100, avg_train_loss))

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for carrier_sig, params, modulated_sig in tqdm(val_loader):
                param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
                input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
                target_seq_batch = modulated_sig.to(device)
                batch_loss = train_step(input_seq_batch, target_seq_batch, model, optimizer, criterion, validate=True)
                val_loss += batch_loss
        avg_val_loss = val_loss / len(val_loader)

        loss_history.append([avg_train_loss, avg_val_loss])
    return loss_history


def train_step(input_tensor, target_tensor, model, optimizer, criterion, validate=False):
    output = model(input_tensor).unsqueeze(-1)
    loss = criterion(output, target_tensor)
    if not validate:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def main(args):
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 30
    train(args.dataset_path, batch_size, num_epochs, learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    main(parser.parse_args())
