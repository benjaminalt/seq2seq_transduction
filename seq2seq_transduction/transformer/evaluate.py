import argparse
import os

import torch
from torch.utils.data import TensorDataset, DataLoader

from seq2seq_transduction.data import load_dataset
from tst import Transformer
from seq2seq_transduction.utils import plot_waves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, input_sequences):
    with torch.no_grad():
        return model(input_sequences)


def main(args):
    normalized_carrier, normalized_params, normalized_modulated, _, _ = load_dataset(args.data_dir, "test")
    checkpoint = torch.load(os.path.join(args.model_file))
    p = checkpoint["parameters"]
    model = Transformer(p["d_input"], p["d_model"], p["d_output"], p["q"], p["v"], p["h"], p["N"], p["attention_size"],
                        p["dropout"], p["chunk_mode"], p["pe"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if args.command == "batch":
        dataset = TensorDataset(torch.from_numpy(normalized_carrier).float(),
                                torch.from_numpy(normalized_params).float(),
                                torch.from_numpy(normalized_modulated).float())
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1,
                                 pin_memory=True, drop_last=True)
        loss_fn = torch.nn.MSELoss()
        loss = 0
        for carrier_sig, params, modulated_sig in data_loader:
            param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
            input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
            target_seq_batch = modulated_sig.to(device)
            output_seq_batch = evaluate(model, input_seq_batch)
            loss += loss_fn(output_seq_batch, target_seq_batch).item()
        loss /= len(data_loader)
        print("Test loss: {}".format(loss))
    elif args.command == "plot":
        for i in range(len(normalized_params)):
            carrier_sig = torch.from_numpy(normalized_carrier[i]).float().unsqueeze(0).to(device)
            params = torch.from_numpy(normalized_params[i]).float().unsqueeze(0).to(device)
            param_seq_batch = params.unsqueeze(1).repeat(1, carrier_sig.size(1), 1)
            input_seq_batch = torch.cat((param_seq_batch, carrier_sig), dim=-1).to(device)
            target = torch.from_numpy(normalized_modulated[i]).float().unsqueeze(0).to(device)
            output = evaluate(model, input_seq_batch)
            plot_waves(params, output.detach().cpu().numpy(), target.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plot", "batch"])
    parser.add_argument("model_file", type=str)
    parser.add_argument("data_dir", type=str)
    main(parser.parse_args())
