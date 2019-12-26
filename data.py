import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

amplitude_limits = [0.1, 5]
freq_limits = [0.1, 5]
phase_limits = [0, 2 * np.pi]


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


def main():
    params, data = generate_sins(5, 100)
    normalized_params, _ = normalize(params)
    normalized_data, _ = normalize(data)
    fig, ax = plt.subplots(2)
    for i in range(5):
        param_set = params[i]
        param_set_normalized = normalized_params[i]
        y = data[i]
        y_normalized = normalized_data[i]
        ax[0].plot(np.arange(y.shape[0]), y, label="A: {:.2f}, f: {:.2f}, phi: {:.2f}".format(*param_set))
        ax[1].plot(np.arange(y_normalized.shape[0]), y_normalized, label="A: {:.2f}, f: {:.2f}, phi: {:.2f}".format(*param_set_normalized))
    ax[0].set_title("Raw")
    ax[1].set_title("Normalized")
    ax[0].legend()
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
