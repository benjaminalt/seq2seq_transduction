import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader

from data import load_dataset
from encoder_decoder.model import Seq2SeqModel
from utils import plot_waves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, input_sequences):
    with torch.no_grad():
        batch_size, seq_len, input_size = input_sequences.size()

        encoder_hidden = model.encoder.init_hidden(batch_size=batch_size, device=device)
        encoder_outputs = torch.zeros(1, seq_len, model.encoder.hidden_size, device=device)

        for ei in range(seq_len):
            encoder_output, encoder_hidden = model.encoder(input_sequences[:, ei, :].unsqueeze(1), encoder_hidden)
            encoder_outputs[:, ei] = encoder_output[:, 0]

        decoder_input = torch.zeros(batch_size, 1, model.decoder.output_size, device=device)
        decoder_hidden = encoder_hidden
        decoder_output = None

        for di in range(seq_len):
            new_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            if decoder_output is None:
                decoder_output = new_output.detach()
            else:
                decoder_output = torch.cat((decoder_output, new_output[:,-1,:].detach().unsqueeze(1)), dim=1)
            decoder_input = decoder_output

        return decoder_output


def main(args):
    normalized_carrier, normalized_params, normalized_modulated, _, _ = load_dataset(args.data_dir, "test")
    model = Seq2SeqModel.load(args.model_dir, device)
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
    parser.add_argument("model_dir", type=str)
    parser.add_argument("data_dir", type=str)
    main(parser.parse_args())
