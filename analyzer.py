import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.stats import ttest_ind, mannwhitneyu
import scipy.stats as stats
import pandas as pd
class TimeDomainAnalyzer:
    """
    # Example usage
    time_series_1 = np.random.randn(600_000)  # Example ts_1 data (replace with your own)
    time_series_2 = np.random.randn(600_000)  # Example ts_2 data (replace with your own)
    sampling_rate = 10_000  # Example sampling rate (replace with your own)

    analyzer = TimeDomainAnalyzer(time_series_1, time_series_2, sampling_rate)
    mean_result = analyzer.compare_mean()
    print("mean comparison:", mean_result)
    ho_result = analyzer.compare_higher_moments(2)
    print("higher order comparison:", ho_result)
    analyzer.plot_timeseries()
    t_stat, p_value = analyzer.perform_hypothesis_test()
    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_value}")
    """
    def __init__(self, time_series_1, time_series_2, sampling_rate):
        self.time_series_1 = time_series_1
        self.time_series_2 = time_series_2
        self.sampling_rate = sampling_rate

    def plot_timeseries(self):
        time = np.arange(len(self.time_series_1)) / self.sampling_rate

        sns.set(style="whitegrid")
        plt.plot(time, self.time_series_1, label='ts_1 Data')
        plt.plot(time, self.time_series_2, label='ts_2 Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def compare_mean(self):
        ts_1_mean = np.mean(self.time_series_1)
        ts_2_mean = np.mean(self.time_series_2)

        comparison_result = np.abs(ts_1_mean - ts_2_mean)
        return comparison_result

    def compare_variance(self):
        ts_1_var = np.var(self.time_series_1)
        ts_2_var = np.var(self.time_series_2)

        comparison_result = np.abs(ts_1_var - ts_2_var)
        return comparison_result

    def compare_std(self):
        ts_1_std = np.std(self.time_series_1)
        ts_2_std = np.std(self.time_series_2)

        comparison_result = np.abs(ts_1_std - ts_2_std)
        return comparison_result

    def compare_skewness(self):
        ts_1_skewness = np.skew(self.time_series_1)
        ts_2_skewness = np.skew(self.time_series_2)

        comparison_result = np.abs(ts_1_skewness - ts_2_skewness)
        return comparison_result

    def compare_kurtosis(self):
        ts_1_kurtosis = np.kurtosis(self.time_series_1)
        ts_2_kurtosis = np.kurtosis(self.time_series_2)

        comparison_result = np.abs(ts_1_kurtosis - ts_2_kurtosis)
        return comparison_result

    def compare_higher_moments(self, order):
        ts_1_moment = np.mean((self.time_series_1 - np.mean(self.time_series_1)) ** order)
        ts_2_moment = np.mean((self.time_series_2 - np.mean(self.time_series_2)) ** order)

        comparison_result = np.abs(ts_1_moment - ts_2_moment)
        return comparison_result

    def plot_distributions(self):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        sns.histplot(self.time_series_1, ax=axes[0], kde=True, color='skyblue')
        axes[0].set_title('ts_1 Data Distribution')

        sns.histplot(self.time_series_2, ax=axes[1], kde=True, color='salmon')
        axes[1].set_title('ts_2 Data Distribution')

        plt.tight_layout()
        plt.show()

    def plot_distributions_one_plot(self):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.histplot(self.time_series_1, ax=ax, kde=True, color='skyblue', label='ts_1 Data')
        sns.histplot(self.time_series_2, ax=ax, kde=True, color='salmon', label='ts_2 Data')

        ax.set_title('Distribution Comparison')
        ax.legend()
        plt.show()

    def plot_boxplots(self):
        sns.set(style="whitegrid")
        fig, ax = plt.subplots()

        # Combine the data into a list for box plot
        data = [self.time_series_1, self.time_series_2]

        # Plot the box plots
        ax.boxplot(data, labels=['ts_1 Data', 'ts_2 Data'])

        # Set labels and title
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Values')
        ax.set_title('Box Plots of ts_1 and ts_2 Data')

        # Show the plot
        plt.tight_layout()
        plt.show()


    def plot_boxplots_sns(self):
        sns.set(style="whitegrid")
        data = {
            'Dataset': ['ts_1 Data'] * len(self.time_series_1) + ['ts_2 Data'] * len(self.time_series_2),
            'Values': np.concatenate((self.time_series_1, self.time_series_2))
        }
        df = pd.DataFrame(data)

        # Create the box plot using seaborn
        sns.boxplot(x='Dataset', y='Values', data=df)

        # Set labels and title
        plt.xlabel('Dataset')
        plt.ylabel('Values')
        plt.title('Box Plots of ts_1 and ts_2 Data')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def plot_autocorrelation(self):
        autocorr_ts_1 = correlate(self.time_series_1, self.time_series_1, mode='same') / len(self.time_series_1)
        autocorr_ts_2 = correlate(self.time_series_2, self.time_series_2, mode='same') / len(self.time_series_2)

        time = np.arange(-len(self.time_series_1) // 2, len(self.time_series_1) // 2) / self.sampling_rate

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(time, autocorr_ts_1, label='ts_1 Data')
        ax.plot(time, autocorr_ts_2, label='ts_2 Data')
        ax.set_xlabel('Time Lag (s)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation Analysis')
        ax.legend()
        plt.tight_layout()
        plt.show()


    def perform_correlation_analysis(self):
        # Calculate Pearson correlation coefficient and p-value
        correlation_coeff, p_value = stats.pearsonr(self.time_series_1, self.time_series_2)

        print(f"Pearson correlation coefficient: {correlation_coeff}")
        print(f"P-value: {p_value}")

        if p_value < 0.05:
            print("Reject Null Hypothesis: There is a significant correlation between the time series. Indicating similarity between the time series.")
        else:
            print("Accept Null Hypothesis: There is no significant correlation between the time series. Indicating differences between the time series.")

    def perform_cross_correlation(self):
        cross_corr = np.correlate(self.time_series_1, self.time_series_2, mode='full')
        lags = np.arange(-len(self.time_series_1) + 1, len(self.time_series_2))

        max_corr = np.max(cross_corr)
        max_corr_index = np.argmax(cross_corr)

        print(f"Max Cross-Correlation: {max_corr} at lag: {lags[max_corr_index]}")


    def compute_cross_correlation(self, max_lag=None, significance_level=0.05):
        import numpy as np
        from scipy.signal import correlate

        cross_corr = correlate(self.time_series_1, self.time_series_2, mode='full')
        if max_lag is not None:
            cross_corr = cross_corr[len(cross_corr) // 2 - max_lag:len(cross_corr) // 2 + max_lag + 1]

        # Compute the lags
        lags = np.arange(-len(cross_corr) // 2 + 1, len(cross_corr) // 2 + 1)

        max_corr_idx = np.argmax(np.abs(cross_corr))
        max_corr_value = cross_corr[max_corr_idx]

        lag = lags[max_corr_idx]

        n = len(self.time_series_1)
        degrees_of_freedom = n - 2  # Assuming two time series

        # Compute p-value using t-distribution
        t_value = max_corr_value * np.sqrt(degrees_of_freedom / (n - max_corr_idx - 1))
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_value), df=degrees_of_freedom))

        is_significant = p_value < significance_level

        print(f"Max Cross-Correlation: {max_corr_value} at lag: {lag}. Is significant: {is_significant}")

        return max_corr_value, lag, cross_corr, lags, p_value, is_significant


    def plot_cross_correlation(self, lags, cross_correlation):
        import matplotlib.pyplot as plt
        plt.plot(lags, cross_correlation)
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.title('Cross-Correlation between Time Series 1 and Time Series 2')
        plt.show()

    def compute_dtw_similarity(self):
        from fastdtw import fastdtw

        # Perform Dynamic Time Warping
        distance, path = fastdtw(self.time_series_1, self.time_series_2)

        # Compute similarity score as the inverse of the distance
        similarity = 1 / distance
        print(f"Similarity: {similarity} \tDistance: {distance}")
        return similarity, path

    def plot_dtw_alignment(self, alignment_path):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        ax.plot(self.time_series_1, label='Time Series 1')
        ax.plot(self.time_series_2, label='Time Series 2')

        # Plot the DTW path
        path_x = [point[0] for point in alignment_path]
        path_y = [point[1] for point in alignment_path]
        ax.plot(path_x, path_y, color='red', linewidth=2, label='DTW Path')

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('DTW Alignment')

        ax.legend()
        plt.show()


    def compute_compression_dissimilarity(self):
        import zlib
        # Convert time series to bytes
        bytes_1 = self.time_series_1.tobytes()
        bytes_2 = self.time_series_2.tobytes()

        # Compute compressed sizes
        compressed_size_1 = len(zlib.compress(bytes_1))
        compressed_size_2 = len(zlib.compress(bytes_2))

        # Compute dissimilarity as the absolute difference in compressed sizes
        dissimilarity = abs(compressed_size_1 - compressed_size_2)
        print(f"Compression-based dissimilarity: {dissimilarity} \t 0 means similarity")
        return dissimilarity


    def calculate_frequencies(self, num_bins=10):
        import numpy as np
        # Determine the range of values
        min_value = min(np.min(self.time_series_1), np.min(self.time_series_2))
        max_value = max(np.max(self.time_series_1), np.max(self.time_series_2))

        # Define equally spaced bins
        bins = np.linspace(min_value, max_value, num_bins + 1)

        # Calculate observed frequencies for each time series
        observed_freq_1, _ = np.histogram(self.time_series_1, bins=bins)
        observed_freq_2, _ = np.histogram(self.time_series_2, bins=bins)

        # Calculate expected frequencies assuming independence
        expected_freq = (observed_freq_1 + observed_freq_2) / 2

        return observed_freq_1, observed_freq_2, expected_freq

    def perform_hypothesis_test(self):
        # Perform a hypothesis test to compare the means of the two datasets
        t_stat, p_value = ttest_ind(self.time_series_1, self.time_series_2)

        # Return the test statistic and p-value
        return t_stat, p_value

    def perform_mann_whitney_test(self):
        # Perform a hypothesis test to compare the distributions of the two datasets using Mann-Whitney U test
        _, p_value = mannwhitneyu(self.time_series_1, self.time_series_2, alternative='two-sided')

        # Return the p-value
        return p_value

    def compare_distributions_mannwithneyu(self):
        stat, p_value = mannwhitneyu(self.time_series_1, self.time_series_2, alternative='two-sided')

        if p_value < 0.05:
            print("Reject Null Hypothesis: The distributions are significantly different.")
        else:
            print("Accept Null Hypothesis: The distributions are not significantly different.")

        print(f"Mann-Whitney U statistic: {stat}")
        print(f"P-value: {p_value}")

    def compare_distributions_chisquare(self):
        observed_freq_1, observed_freq_2, expected_freq = self.calculate_frequencies(num_bins=1000)
        stat, p_value = stats.chisquare(observed_freq_1, f_exp=expected_freq)
        #stat, p_value = stats.chisquare(self.time_series_1, self.time_series_2)

        if p_value < 0.05:
            print("Reject Null Hypothesis: The distributions are significantly different.")
        else:
            print("Accept Null Hypothesis: The distributions are not significantly different.")

        print(f"Chi² statistic: {stat}")
        print(f"P-value: {p_value}")


    def test_variances(self):
        _, p_value = stats.bartlett(self.time_series_1, self.time_series_2)
        # Alternatively, you can use Levene's test:
        #_, p_value = stats.levene(self.time_series_1, self.time_series_2)

        if p_value < 0.05:
            print("The variances are significantly different.")
        else:
            print("The variances are not significantly different.")

        print(f"P-value: {p_value}")

    def test_normality_shapirowilk(self, data):
        stat, p_value = stats.shapiro(data)

        if p_value < 0.05:
            print("The data does not follow a normal distribution.")
        else:
            print("The data follows a normal distribution.")

        print(f"Shapiro-Wilk test statistic: {stat}")
        print(f"P-value: {p_value}")

    def test_normality_anderson(self, data):
        result = stats.anderson(data)

        if result.statistic < result.critical_values[2]:
            print("The data follows a normal distribution.")
        else:
            print("The data does not follow a normal distribution.")

        print(f"Anderson-Darling test statistic: {result.statistic}")
        print(f"Critical values: {result.critical_values}")

    def test_normality_kstest(self, data):
        result = stats.kstest(data, 'norm')

        if result.pvalue < 0.05:
            print("The data does not follow a normal distribution.")
        else:
            print("The data follows a normal distribution.")

        print(f"Kolmogorov-Smirnov test statistic: {result.statistic}")
        print(f"P-value: {result.pvalue}")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class FrequencyDomainAnalyzer:
    """
    # Example usage
    import numpy as np
    time_series_1 = np.random.randn(600_000)  # Example ts_1 data (replace with your own)
    time_series_2 = np.random.randn(600_000)  # Example ts_2 data (replace with your own)
    sampling_rate = 10_000

    analyzer = FrequencyDomainAnalyzer(time_series_1, time_series_2, sampling_rate)
    comparison_result = analyzer.compare_psd()
    analyzer.plot_fft(analyzer.time_series_1)

    print(f"Comparison result: {comparison_result}")
    """
    def __init__(self, time_series_1, time_series_2, sampling_rate):
        self.time_series_1 = time_series_1
        self.time_series_2 = time_series_2
        self.sampling_rate = sampling_rate

    def compute_fft(self, data):
        fft_result = np.fft.fft(data)
        return fft_result

    def compute_psd(self, fft_result):
        psd_result = np.abs(fft_result) ** 2
        return psd_result

    def plot_psd(self, ts_1_psd, ts_2_psd):
        frequency = np.fft.fftfreq(len(ts_1_psd))
        sns.set(style="whitegrid")
        plt.plot(frequency, ts_1_psd, label='ts_1 Data')
        plt.plot(frequency, ts_2_psd, label='ts_2 Data')
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
        ts_1_fft = self.compute_fft(self.time_series_1)
        ts_2_fft = self.compute_fft(self.time_series_2)

        ts_1_psd = self.compute_psd(ts_1_fft)
        ts_2_psd = self.compute_psd(ts_2_fft)

        self.plot_psd(ts_1_psd, ts_2_psd)

        # Compare the PSDs using a suitable metric (e.g., mean squared error, correlation, etc.)
        # TODO For MSE: Normalizing PSD output (linear, logarithmic, z-score) before putting into MSE
        comparison_result = np.mean((ts_1_psd - ts_2_psd) ** 2)

        return comparison_result


import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataNormalizer:
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = None

    def fit(self, data):
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Invalid normalization method. Choose 'standard' or 'minmax'.")

        self.scaler.fit(data)

    def transform(self, data):
        if self.scaler is None:
            raise ValueError("Fit the normalizer first by calling the 'fit' method.")

        normalized_data = self.scaler.transform(data)
        return normalized_data

    def fit_transform(self, data):
        self.fit(data)
        normalized_data = self.transform(data)
        return normalized_data





