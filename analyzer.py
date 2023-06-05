import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.stats import ttest_ind, mannwhitneyu

class TimeDomainAnalyzer:
    """
    # Example usage
    insilico_data = np.random.randn(600_000)  # Example insilico data (replace with your own)
    real_world_data = np.random.randn(600_000)  # Example real-world data (replace with your own)
    sampling_rate = 10_000  # Example sampling rate (replace with your own)

    analyzer = TimeDomainAnalyzer(insilico_data, real_world_data, sampling_rate)
    mean_result = analyzer.compare_mean()
    print("mean comparison:", mean_result)
    ho_result = analyzer.compare_higher_moments(2)
    print("higher order comparison:", ho_result)
    analyzer.plot_timeseries()
    t_stat, p_value = analyzer.perform_hypothesis_test()
    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_value}")
    """
    def __init__(self, insilico_data, real_world_data, sampling_rate):
        self.insilico_data = insilico_data
        self.real_world_data = real_world_data
        self.sampling_rate = sampling_rate

    def plot_timeseries(self):
        time = np.arange(len(self.insilico_data)) / self.sampling_rate

        sns.set(style="whitegrid")
        plt.plot(time, self.insilico_data, label='Insilico Data')
        plt.plot(time, self.real_world_data, label='Real World Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_mean(self):
        insilico_mean = np.mean(self.insilico_data)
        real_world_mean = np.mean(self.real_world_data)

        comparison_result = np.abs(insilico_mean - real_world_mean)
        return comparison_result

    def compare_variance(self):
        insilico_var = np.var(self.insilico_data)
        real_world_var = np.var(self.real_world_data)

        comparison_result = np.abs(insilico_var - real_world_var)
        return comparison_result

    def compare_skewness(self):
        insilico_skewness = np.skew(self.insilico_data)
        real_world_skewness = np.skew(self.real_world_data)

        comparison_result = np.abs(insilico_skewness - real_world_skewness)
        return comparison_result

    def compare_kurtosis(self):
        insilico_kurtosis = np.kurtosis(self.insilico_data)
        real_world_kurtosis = np.kurtosis(self.real_world_data)

        comparison_result = np.abs(insilico_kurtosis - real_world_kurtosis)
        return comparison_result

    def compare_higher_moments(self, order):
        insilico_moment = np.mean((self.insilico_data - np.mean(self.insilico_data)) ** order)
        real_world_moment = np.mean((self.real_world_data - np.mean(self.real_world_data)) ** order)

        comparison_result = np.abs(insilico_moment - real_world_moment)
        return comparison_result

    def plot_distributions(self):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        sns.histplot(self.insilico_data, ax=axes[0], kde=True, color='skyblue')
        axes[0].set_title('Insilico Data Distribution')

        sns.histplot(self.real_world_data, ax=axes[1], kde=True, color='salmon')
        axes[1].set_title('Real World Data Distribution')

        plt.tight_layout()
        plt.show()

    def plot_distributions_one_plot(self):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.histplot(self.insilico_data, ax=ax, kde=True, color='skyblue', label='Insilico Data')
        sns.histplot(self.real_world_data, ax=ax, kde=True, color='salmon', label='Real World Data')

        ax.set_title('Distribution Comparison')
        ax.legend()
        plt.show()


    def plot_autocorrelation(self):
        autocorr_insilico = correlate(self.insilico_data, self.insilico_data, mode='same') / len(self.insilico_data)
        autocorr_real_world = correlate(self.real_world_data, self.real_world_data, mode='same') / len(self.real_world_data)

        time = np.arange(-len(self.insilico_data) // 2, len(self.insilico_data) // 2) / self.sampling_rate

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(time, autocorr_insilico, label='Insilico Data')
        ax.plot(time, autocorr_real_world, label='Real World Data')
        ax.set_xlabel('Time Lag (s)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation Analysis')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def perform_hypothesis_test(self):
        # Perform a hypothesis test to compare the means of the two datasets
        t_stat, p_value = ttest_ind(self.insilico_data, self.real_world_data)

        # Return the test statistic and p-value
        return t_stat, p_value

    def perform_mann_whitney_test(self):
        # Perform a hypothesis test to compare the distributions of the two datasets using Mann-Whitney U test
        _, p_value = mannwhitneyu(self.insilico_data, self.real_world_data, alternative='two-sided')

        # Return the p-value
        return p_value


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class FrequencyDomainAnalyzer:
    """
    # Example usage
    import numpy as np
    insilico_data = np.random.randn(600_000)  # Example insilico data (replace with your own)
    real_world_data = np.random.randn(600_000)  # Example real-world data (replace with your own)
    sampling_rate = 10_000

    analyzer = FrequencyDomainAnalyzer(insilico_data, real_world_data, sampling_rate)
    comparison_result = analyzer.compare_psd()
    analyzer.plot_fft(analyzer.insilico_data)

    print(f"Comparison result: {comparison_result}")
    """
    def __init__(self, insilico_data, real_world_data, sampling_rate):
        self.insilico_data = insilico_data
        self.real_world_data = real_world_data
        self.sampling_rate = sampling_rate

    def compute_fft(self, data):
        fft_result = np.fft.fft(data)
        return fft_result

    def compute_psd(self, fft_result):
        psd_result = np.abs(fft_result) ** 2
        return psd_result

    def plot_psd(self, insilico_psd, real_world_psd):
        frequency = np.fft.fftfreq(len(insilico_psd))
        sns.set(style="whitegrid")
        plt.plot(frequency, insilico_psd, label='Insilico Data')
        plt.plot(frequency, real_world_psd, label='Real World Data')
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectral Density')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_fft(self, data):
        fft_result = self.compute_fft(data)
        frequency = np.fft.fftfreq(len(data), 1 / self.sampling_rate)

        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)

        sns.set(style="whitegrid")
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        axs[0].plot(frequency, magnitude)
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Magnitude')

        axs[1].plot(frequency, phase)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Phase')

        plt.tight_layout()
        plt.show()

    def compare_psd(self):
        insilico_fft = self.compute_fft(self.insilico_data)
        real_world_fft = self.compute_fft(self.real_world_data)

        insilico_psd = self.compute_psd(insilico_fft)
        real_world_psd = self.compute_psd(real_world_fft)

        self.plot_psd(insilico_psd, real_world_psd)

        # Compare the PSDs using a suitable metric (e.g., mean squared error, correlation, etc.)
        # TODO For MSE: Normalizing PSD output (linear, logarithmic, z-score) before putting into MSE
        comparison_result = np.mean((insilico_psd - real_world_psd) ** 2)

        return comparison_result
