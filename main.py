

if __name__ == '__main__':

    # Example usage
    import numpy as np
    insilico_data = np.random.randn(600_000)  # Example insilico data (replace with your own)
    real_world_data = np.random.randn(600_000)  # Example real-world data (replace with your own)
    sampling_rate = 10_000  # Example sampling rate (replace with your own)

    from analyzer import FrequencyDomainAnalyzer

    analyzer = FrequencyDomainAnalyzer(insilico_data, real_world_data, sampling_rate)
    #analyzer.plot_fft(analyzer.insilico_data)
    analyzer.compare_psd()