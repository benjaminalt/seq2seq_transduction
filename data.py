import argparse
import os

import joblib
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import plot_waves

amplitude_limits = [0.1, 5]
freq_limits = [0.1, 5]
phase_limits = [0, 2 * np.pi]

script_dir = os.path.abspath(os.path.dirname(__file__))


def generate_sins(n, seq_len):
    params = []
    data = []
    for _ in range(n):
        amp = random.uniform(*amplitude_limits)
        freq = random.uniform(*freq_limits)
        phase = random.uniform(*phase_limits)
        x = np.arange(seq_len)
        y = amp * np.sin(2 * np.pi * freq * (x / seq_len) + phase)
        params.append([amp, freq, phase])
        data.append(y)
    return np.array(params), np.expand_dims(np.array(data), -1)


def amplitude_modulate(carrier, amplitudes):
    return carrier * amplitudes


def normalize(data, scaler=None):
    orig_shape = data.shape
    if len(data.shape) == 3:
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaler.fit(data)
    normalized_data = scaler.transform(data).reshape(orig_shape)
    return normalized_data, scaler


def denormalize(data, scaler):
    orig_shape = data.shape
    if len(data.shape) == 3:
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    denormalized_data = scaler.inverse_transform(data).reshape(orig_shape)
    return denormalized_data


def generate_dataset(n, seq_len, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        for kind in ("train", "validate", "test"):
            os.makedirs(os.path.join(output_dir, kind))

    print("Generating dataset in {}...".format(output_dir))
    _, carrier_signals = generate_sins(n, seq_len)
    params, amplitudes = generate_sins(n, seq_len)
    modulated_signals = amplitude_modulate(carrier_signals, amplitudes)
    normalized_carrier, signal_scaler = normalize(carrier_signals)
    normalized_params, param_scaler = normalize(params)
    normalized_modulated, _ = normalize(modulated_signals, signal_scaler)

    np.save(os.path.join(output_dir, "train", "carrier.npy"), normalized_carrier)
    np.save(os.path.join(output_dir, "train", "params.npy"), normalized_params)
    np.save(os.path.join(output_dir, "train", "modulated.npy"), normalized_modulated)

    for kind in ("validate", "test"):
        num_data = int(0.2 * n)
        _, carrier_signals = generate_sins(num_data, seq_len)
        params, amplitudes = generate_sins(num_data, seq_len)
        modulated_signals = amplitude_modulate(carrier_signals, amplitudes)
        normalized_carrier, _ = normalize(carrier_signals, signal_scaler)
        normalized_params, _ = normalize(params, param_scaler)
        normalized_modulated, _ = normalize(modulated_signals, signal_scaler)
        np.save(os.path.join(output_dir, kind, "carrier.npy"), normalized_carrier)
        np.save(os.path.join(output_dir, kind, "params.npy"), normalized_params)
        np.save(os.path.join(output_dir, kind, "modulated.npy"), normalized_modulated)

    joblib.dump(signal_scaler, os.path.join(output_dir, "signal_scaler.pkl"))
    joblib.dump(param_scaler, os.path.join(output_dir, "param_scaler.pkl"))
    print("Done.")


def load_dataset(dataset_path, kind="train"):
    print("Loading dataset from {}...".format(dataset_path))
    normalized_carrier = np.load(os.path.join(dataset_path, kind, "carrier.npy"))
    normalized_params = np.load(os.path.join(dataset_path, kind, "params.npy"))
    normalized_modulated = np.load(os.path.join(dataset_path, kind, "modulated.npy"))
    signal_scaler = joblib.load(os.path.join(dataset_path, "signal_scaler.pkl"))
    param_scaler = joblib.load(os.path.join(dataset_path, "param_scaler.pkl"))
    print("Done.")
    return normalized_carrier, normalized_params, normalized_modulated, signal_scaler, param_scaler


def main(args):
    if args.command == "plot":
        params, data = generate_sins(args.n, args.seq_len)
        normalized_params, _ = normalize(params)
        normalized_data, _ = normalize(data)
        plot_waves(normalized_params, normalized_data)
    elif args.command == "generate_dataset":
        generate_dataset(args.n, args.seq_len, args.output_dir)
    else:
        raise ValueError("Invalid command: {}".format(args.command))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plot", "generate_dataset"])
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "data"))
    main(parser.parse_args())
