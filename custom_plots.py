def plot_gt_signal_comparison(signal_raw, signal_raw_noise, timestamps):
    import numpy as np
    import matplotlib.pyplot as plt

    signal_sub = np.subtract(signal_raw, signal_raw_noise)

    i = 6
    plt.plot(timestamps, signal_raw[:,i], label='Signal + Noise', color='blue')
    plt.plot(timestamps, signal_raw_noise[:,i], label='Noise', color='deepskyblue')
    plt.plot(timestamps, signal_sub[:,i], label='Signal - Noise', color='gray')
    plt.title('Signal and noise comparison (fs=8000)')
    plt.xlabel('Time in sec')
    plt.ylabel('Amplitude in µV')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_psd_comparison_methods(signal_x = real_world_data_noise, signal_y = real_world_data_spikes, t = timestamps, fs = 8000):
    import matplotlib.pyplot as plt
    from scipy import signal
    f_x, Pxx_den_x = signal.welch(x=signal_x, fs=fs)
    f_y, Pxx_den_y = signal.welch(x=signal_y, fs=fs)

    fig, ax = plt.subplots(2,2)
    ax[0][0].plot(t, signal_x, color='blue')
    ax[0][0].set_title('Real World Noise')
    ax[0][0].set_ylim(-300, 300)
    ax[0][0].set_xlabel('time [s]')
    ax[0][0].set_ylabel('Amplitude [µV]')

    ax[0][1].plot(t, signal_y, color='dodgerblue')
    ax[0][1].set_title('Real World Spikes')
    ax[0][1].set_ylim(-300, 300)
    ax[0][1].set_xlabel('time [s]')
    ax[0][1].set_ylabel('Amplitude [µV]')

    ax[1][0].semilogy(f_x, Pxx_den_x, color='blue')
    ax[1][0].semilogy(f_y, Pxx_den_y, color='dodgerblue')
    ax[1][0].set_title('PSD (Scipy)')
    ax[1][0].set_xlabel('frequency [Hz]')
    ax[1][0].set_ylabel('PSD [V**2/Hz]')
    ax[1][0].grid(True)

    ax[1][1].psd(x=signal_x, Fs=fs, color='blue')
    ax[1][1].psd(x=signal_y, Fs=fs, color='dodgerblue')
    ax[1][1].set_title('PSD (Matplotlib)')

    fig.tight_layout(pad=1.0)
    plt.show()


def plot_csd_and_coherence_comparison_methods(signal_x = real_world_data_spikes, signal_y = insilico_data_spikes, t = timestamps, fs = 8000):
    import matplotlib.pyplot as plt
    from scipy import signal
    f_xy, Pxy = signal.csd(x=signal_x, y=signal_y, fs=fs)
    f_xy, Cxy = signal.coherence(x=signal_x, y=signal_y, fs=fs)

    fig, ax = plt.subplots(nrows=3,ncols=2)

    ax[0][0].plot(t, signal_x, color='blue')
    ax[0][0].set_title('Real World')
    ax[0][0].set_ylim(-300, 300)
    ax[0][0].set_xlabel('time [s]')
    ax[0][0].set_ylabel('Amplitude [µV]')

    ax[0][1].plot(t, signal_y, color='blue')
    ax[0][1].set_title('Insilico')
    ax[0][1].set_ylim(-300, 300)
    ax[0][1].set_xlabel('time [s]')
    ax[0][1].set_ylabel('Amplitude [µV]')

    ax[1][0].semilogy(f_xy, Pxy, color='blue')
    ax[1][0].set_title('CSD (Scipy)')
    ax[1][0].set_xlabel('frequency [Hz]')
    ax[1][0].set_ylabel('CSD [V**2/Hz]')
    ax[1][0].grid(True)

    ax[1][1].csd(x=signal_x, y=signal_y, Fs=fs, color='blue')
    ax[1][1].set_title('CSD (Matplotlib)')

    f_xy, Cxy = signal.coherence(x=signal_x, y=signal_y, fs=fs)

    ax[2][0].semilogy(f_xy, Cxy, color='blue')
    ax[2][0].set_title('Coherence (Scipy)')
    ax[2][0].set_xlabel('frequency [Hz]')
    ax[2][0].set_ylabel('Coherence')
    ax[2][0].grid(True)

    ax[2][1].cohere(x=signal_x, y=signal_y, Fs=fs, color='blue')
    ax[2][1].set_title('Coherence (Matplotlib)')

    fig.tight_layout(pad=1.0)
    plt.show()


def plot_spectrogram_comparison_methods(real_world_data_spikes, real_world_data_noise, insilico_data_spikes, insilico_data_noise, fs=8000):
    import matplotlib.pyplot as plt
    from scipy import signal

    fig, ax = plt.subplots(nrows=4,ncols=2)

    f, t, Sxx = signal.spectrogram(x=real_world_data_spikes, fs=fs)
    ax[0][0].pcolormesh(t, f, Sxx)#, shading='gouraud')
    ax[0][0].set_title('Real World Spikes')
    ax[0][0].set_ylabel('Frequency [Hz]')
    ax[0][0].set_xlabel('Time [sec]')

    f, t, Sxx = signal.spectrogram(x=real_world_data_noise, fs=fs)
    ax[1][0].pcolormesh(t, f, Sxx)#, shading='gouraud')
    ax[1][0].set_title('Real World Noise')
    ax[1][0].set_ylabel('Frequency [Hz]')
    ax[1][0].set_xlabel('Time [sec]')

    f, t, Sxx = signal.spectrogram(x=insilico_data_spikes, fs=fs)
    ax[2][0].pcolormesh(t, f, Sxx)#, shading='gouraud')
    ax[2][0].set_title('Insilico Spikes')
    ax[2][0].set_ylabel('Frequency [Hz]')
    ax[2][0].set_xlabel('Time [sec]')

    f, t, Sxx = signal.spectrogram(x=insilico_data_noise, fs=fs)
    ax[3][0].pcolormesh(t, f, Sxx)#, shading='gouraud')
    ax[3][0].set_title('Insilico Noise')
    ax[3][0].set_ylabel('Frequency [Hz]')
    ax[3][0].set_xlabel('Time [sec]')

    ax[0][1].specgram(x=real_world_data_spikes, Fs=fs)
    ax[0][1].set_title('Real World Spikes')

    ax[1][1].specgram(x=real_world_data_noise, Fs=fs)
    ax[1][1].set_title('Real World Noise')

    ax[2][1].specgram(x=insilico_data_spikes, Fs=fs)
    ax[2][1].set_title('Insilico Spikes')

    ax[3][1].specgram(x=insilico_data_noise, Fs=fs)
    ax[3][1].set_title('Insilico Noise')

    fig.tight_layout(pad=0.5)
    fig.suptitle('Comparison Spectrogram Methods (Left: Scipy. Right: Matplotlib)')
    plt.show()


#plot_spectrogram_comparison_methods(real_world_data_spikes, real_world_data_noise, insilico_data_spikes, insilico_data_noise)
#plot_psd_comparison_methods(real_world_data_noise, real_world_data_spikes, timestamps, 8000)
#plot_csd_and_coherence_comparison_methods(real_world_data_spikes, insilico_data_spikes, timestamps, 8000)
