from pathlib import Path

import pandas as pd
import numpy as np
import pywt
from PyEMD import EMD, EEMD
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def check_timeseries_equal(arr1, arr2):
    # Check if arrays have same length
    if len(arr1) != len(arr2):
        return False

    # Use numpy's allclose for float comparison with tolerance
    return np.allclose(arr1, arr2, rtol=1, atol=1)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load carbon price data from CSV file.
    Expected format: datetime index and price column
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df.iloc[:, 0])  # Convert first column to datetime
    df.set_index('Date', inplace=True)
    return df


class WaveletDecomposition:
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        """
        Initialize wavelet decomposition.

        Args:
            wavelet: Wavelet type (default: 'db4')
            level: Decomposition level (default: 3)
        """
        self.wavelet = wavelet
        self.level = level
        self.coefficients = None

    def pad_to_length(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """
        Pad array to target length with zeros.

        Args:
            array: Input array
            target_length: Desired length

        Returns:
            Padded array
        """
        pad_width = target_length - len(array)
        if pad_width > 0:
            return np.pad(array, (0, pad_width), mode='constant')
        return array

    def decompose(self, data: np.ndarray, return_coefs: bool = False) -> Dict:
        """
        Perform wavelet decomposition.

        Args:
            data: 1D numpy array of prices

        Returns:
            Dictionary containing coefficients and details at each level
        """
        # Perform wavelet decomposition
        N = len(data)
        self.coefficients = pywt.wavedec(data, self.wavelet, level=self.level)

        if return_coefs:
            # Return raw coefficients (different lengths)
            decomp = {
                'approximation': self.coefficients[0],
                'details': {}
            }
            print("\nRaw coefficient lengths:")
            print(f"Approximation: {len(self.coefficients[0])}")
            for i in range(1, len(self.coefficients)):
                decomp['details'][f'D{i}'] = self.coefficients[i]
                print(f"Detail D{i}: {len(self.coefficients[i])}")
            return decomp
        else:
            # Reconstruct each level to original signal length
            components = np.zeros((self.level + 1, len(self.coefficients[-1])))

            # Pad to length
            for i in range(len(components)):
                components[i] = self.pad_to_length(self.coefficients[0], len(components[0]))

            print("\nPadded component shapes:")
            print(f"Array shape: {components.shape}")
            print("All components padded to maximum deconstructed length")

            return components

    def reconstruct(self, coefficients: List = None) -> np.ndarray:
        """
        Reconstruct signal from wavelet coefficients.

        Args:
            coefficients: List of coefficients (if None, uses stored coefficients)

        Returns:
            Reconstructed signal
        """
        if coefficients is None:
            coefficients = self.coefficients
        return pywt.waverec(coefficients, self.wavelet)

    def denoise(self, data: np.ndarray, threshold_method: str = 'soft',
                threshold_level: float = None) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.

        Args:
            data: Input signal
            threshold_method: 'soft' or 'hard'
            threshold_level: Custom threshold level (if None, uses universal threshold)

        Returns:
            Denoised signal
        """
        # Decompose
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)

        # Calculate threshold
        if threshold_level is None:
            threshold_level = np.sqrt(2 * np.log(len(data)))

        # Apply thresholding
        denoised_coeffs = []
        for i in range(len(coeffs)):
            if i == 0:  # Skip approximation coefficients
                denoised_coeffs.append(coeffs[i])
            else:
                if threshold_method == 'soft':
                    thresh_coeffs = pywt.threshold(coeffs[i], threshold_level, mode='soft')
                else:
                    thresh_coeffs = pywt.threshold(coeffs[i], threshold_level, mode='hard')
                denoised_coeffs.append(thresh_coeffs)

        # Reconstruct
        return pywt.waverec(denoised_coeffs, self.wavelet)


class EMDDecomposition:
    def __init__(self):
        """Initialize EMD decomposition."""
        self.emd = EMD(spline_kind='akima', trials=30)
        self.imfs = None
        self.residue = None

    def decompose(self, data: np.ndarray) -> Dict:
        """
        Perform EMD decomposition.

        Args:
            data: 1D numpy array of prices

        Returns:
            Dictionary containing IMFs and residue
        """
        # Perform EMD
        self.imfs = self.emd.emd(data, max_imf=3)
        self.residue = data - np.sum(self.imfs, axis=0)

        # Create dictionary to store decomposition
        decomp = {
            'imfs': self.imfs,
            'residue': self.residue
        }

        return decomp

    def reconstruct(self, imfs: np.ndarray = None, residue: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct signal from IMFs and residue.

        Args:
            imfs: Array of IMFs (if None, uses stored IMFs)
            residue: Residue (if None, uses stored residue)

        Returns:
            Reconstructed signal
        """
        if imfs is None:
            imfs = self.imfs
        if residue is None:
            residue = self.residue

        return np.sum(imfs, axis=0) + residue

    def denoise(self, data: np.ndarray, n_imfs: int = None) -> np.ndarray:
        """
        Denoise signal by removing high-frequency IMFs.

        Args:
            data: Input signal
            n_imfs: Number of IMFs to keep (if None, uses half of total IMFs)

        Returns:
            Denoised signal
        """
        # Decompose
        imfs = self.emd.emd(data)

        if n_imfs is None:
            n_imfs = len(imfs) // 2

        # Keep only specified number of IMFs (removing high-frequency components)
        denoised_imfs = imfs[:n_imfs]

        # Reconstruct
        return np.sum(denoised_imfs, axis=0)


def plot_decomposition(original: np.ndarray, decomp: Dict, method: str,
                       dates: pd.DatetimeIndex = None):
    """
    Plot original signal and its decomposition.

    Args:
        original: Original signal
        decomp: Decomposition dictionary
        method: 'wavelet' or 'emd'
        dates: DatetimeIndex for x-axis (optional)
    """
    if method == 'wavelet':
        if isinstance(decomp, np.ndarray):
            n_plots = len(decomp) + 1
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

            # Plot original signal
            axes[0].plot(original)
            axes[0].set_title('Original Signal')

            # Plot approximation
            axes[1].plot(decomp[0])
            axes[1].set_title('Approximation')

            # Plot details
            for i, detail in enumerate(decomp[1:], 2):
                axes[i].plot(detail)
                axes[i].set_title(f'Detail Level D{i - 1}')
        else:
            n_plots = len(decomp['details']) + 2  # +2 for original and approximation
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

            # Plot original signal
            axes[0].plot(original)
            axes[0].set_title('Original Signal')

            # Plot approximation
            axes[1].plot(decomp['approximation'])
            axes[1].set_title('Approximation')

            # Plot details
            for i, (name, detail) in enumerate(decomp['details'].items(), 2):
                axes[i].plot(detail)
                axes[i].set_title(f'Detail Level {name}')

    elif method == 'emd':
        n_plots = len(decomp['imfs']) + 2  # +2 for original and residue
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

        # Plot original signal
        if dates is not None:
            axes[0].plot(dates, original)
        else:
            axes[0].plot(original)
        axes[0].set_title('Original Signal')

        # Plot IMFs
        for i, imf in enumerate(decomp['imfs'], 1):
            if dates is not None:
                axes[i].plot(dates, imf)
            else:
                axes[i].plot(imf)
            axes[i].set_title(f'IMF {i}')

        # Plot residue
        if dates is not None:
            axes[-1].plot(dates, decomp['residue'])
        else:
            axes[-1].plot(decomp['residue'])
        axes[-1].set_title('Residue')

    # plt.tight_layout()
    # plt.show()
    return fig

# Predict with a pre-trained model.
# Training requires separately decomposing each window in the training set, which may be time-consuming. <- not that much tbh, could be a good method?
def predict_w_model(input_data):
    return None


# Example usage:
if __name__ == "__main__":
    # Load data
    df = load_data('C:/Users/stucws/Documents/astar/code/sentiment-analysis/webscrape_headlines/Files Wei Soon/Select feature/daily_prices.csv')
    data = df.iloc[:, 0].values  # Assuming price is in first column

    export_dir = Path('plots/')

    # num_data_points_list = [0, 1024, 512, 256]
    num_data_points_list = [512]

    for num_data_points in num_data_points_list:
        print(f'Running {num_data_points}...')
        shortened_data = data[-num_data_points:]

        # Wavelet decomposition
        wave_decomp = WaveletDecomposition(wavelet='db4', level=3)
        wave_results = wave_decomp.decompose(shortened_data)
        reconstructed_wave = wave_decomp.reconstruct()
        denoised_wave = wave_decomp.denoise(shortened_data)

        # EMD decomposition
        emd_decomp = EMDDecomposition()
        emd_results = emd_decomp.decompose(shortened_data)
        reconstructed_emd = emd_decomp.reconstruct()
        denoised_emd = emd_decomp.denoise(shortened_data)

        # Plot results
        wavelet_plot = plot_decomposition(shortened_data, wave_results, 'wavelet', df.iloc[-num_data_points:].index)
        wavelet_plot.savefig(export_dir / f'wt_decomp_{num_data_points}_samedims.png', bbox_inches='tight')
        emd_plot = plot_decomposition(shortened_data, emd_results, 'emd', df.iloc[-num_data_points:].index)
        emd_plot.savefig(export_dir / f'emd_decomp_{num_data_points}.png', bbox_inches='tight')

        # Plot reconstructions and denoised signals
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(df.iloc[-num_data_points:].index, shortened_data, label='Original')
        plt.plot(df.iloc[-num_data_points:].index, reconstructed_wave, label='Wavelet Reconstruction')
        plt.plot(df.iloc[-num_data_points:].index, reconstructed_emd, label='EMD Reconstruction')
        plt.legend()
        plt.title('Signal Reconstructions')

        plt.subplot(212)
        plt.plot(df.iloc[-num_data_points:].index, shortened_data, label='Original')
        plt.plot(df.iloc[-num_data_points:].index, denoised_wave, label='Wavelet Denoised')
        plt.plot(df.iloc[-num_data_points:].index, denoised_emd, label='EMD Denoised')
        plt.legend()
        plt.title('Denoised Signals')
        plt.tight_layout()
        # plt.show()
        fig.savefig(export_dir / f'reconstruction_{num_data_points}.png', bbox_inches='tight')

        # prediction and reconstruction?
        # NOTE reconstruction requires a prediction horizon of the same length as the decomposed input - may pose an issue!!
        for item in wave_decomp.coefficients:
            print(item.shape)
            predict_w_model(item)

        for item in emd_decomp.imfs:
            print(item.shape)
            predict_w_model(item)

        exit()
