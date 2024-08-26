import torch
import torchaudio
import torchaudio.transforms as T

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import matplotlib

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()

def waveform_to_png(file_path):
    matplotlib.use('Agg')
    y, sr = librosa.load(file_path, sr=None)

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    output_image_path = './static/processed/waveform_image.png'
    plt.savefig(output_image_path)
    plt.close()

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def process_one_shot(file_path):
    # return shape is 1, 257, 8
    waveform, sample_rate = torchaudio.load(file_path)
    waveform_mono = waveform.mean(dim=0, keepdim=True)
    resampled_waveform = torchaudio.functional.resample(waveform_mono, sample_rate, 16000)
    target_length = 2000
    current_length = resampled_waveform.size(1)
    if current_length < target_length:
        padding = target_length - current_length
        padded_waveform = torch.nn.functional.pad(resampled_waveform, (0, padding))
    else:
        padded_waveform = resampled_waveform[:, :target_length]

    spectrogram_transform = T.Spectrogram(n_fft=512)
    spectrogram = spectrogram_transform(padded_waveform)

    return spectrogram

def process_full_loop(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform_mono = waveform.mean(dim=0, keepdim=True)

    resampled_waveform = torchaudio.functional.resample(waveform_mono, sample_rate, 16000)
    waveform_np = resampled_waveform.squeeze().numpy()

    onset_env = librosa.onset.onset_strength(y=waveform_np, sr=16000, hop_length=512)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=16000, 
                                                hop_length=512, backtrack=True, pre_max=5, 
                                                post_max=5, pre_avg=30, post_avg=30, 
                                                delta=0.08, units='frames')

    onset_times = librosa.frames_to_time(onset_frames, sr=16000, hop_length=512)

    if onset_times[-1] < len(waveform_np) / 16000:
        onset_times = np.append(onset_times, len(waveform_np) / 16000)

    spectrograms = []
    for i in range(len(onset_times) - 1):
        start_time = onset_times[i]
        end_time = onset_times[i + 1]
        
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)

        segment = resampled_waveform[:, start_sample:end_sample]

        target_length = 2000
        current_length = segment.size(1)
        if current_length < target_length:
            padding = target_length - current_length
            segment_padded = torch.nn.functional.pad(segment, (0, padding))
        else:
            segment_padded = segment[:, :target_length]

        spectrogram_transform = T.Spectrogram(n_fft=512)
        spectrogram = spectrogram_transform(segment_padded)
        spectrograms.append((onset_times[i], spectrogram))

    return spectrograms

# # example usage
# HIHAT = '../dataset/one_shots/real_drums/hihat/'
# SNARE = '../dataset/one_shots/real_drums/snare/'
# KICK = '../dataset/one_shots/real_drums/kick/'

# all_hihats = {os.path.join(HIHAT, f):0 for f in os.listdir(HIHAT) if os.path.isfile(os.path.join(HIHAT, f))}
# all_snares = {os.path.join(SNARE, f):1 for f in os.listdir(SNARE) if os.path.isfile(os.path.join(SNARE, f))}
# all_kicks = {os.path.join(KICK, f):2 for f in os.listdir(KICK) if os.path.isfile(os.path.join(KICK, f))}
# all_sounds = all_hihats | all_snares | all_kicks

# hh = process_ones_shot(list(all_hihats.keys())[0])
# kick = process_ones_shot(list(all_kicks.keys())[0])
# snare = process_ones_shot(list(all_snares.keys())[0])


# a = process_full_loop('../dataset/full_loops/1.wav')
# x = a[0]
# y =a[1]
# z =a[2]

# fig, axs = plt.subplots(6, 1)
# plot_spectrogram(hh[0], ylabel="hh", ax=axs[0])
# plot_spectrogram(y[0], ylabel="hh loop", ax=axs[1])

# plot_spectrogram(kick[0], ylabel="k", ax=axs[2])
# plot_spectrogram(x[0], ylabel="k loop", ax=axs[3])

# plot_spectrogram(snare[0], ylabel="s", ax=axs[4])
# plot_spectrogram(z[0], ylabel="s loop", ax=axs[5])
# plt.show()
