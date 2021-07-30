import numpy as np
from numpy.fft import fft
from numpy.fft import ifft


def stft(x, nperseg=256, nfft=None):
    """
    :param x: input signal
    :param nperseg: length each segment
    :param nfft: length of fft

    :return: stft result
    """
    if nfft is None:
        nfft = nperseg

    x_length = len(x)
    x_padded_length = x_length+2*nperseg

    x_padded = np.zeros(shape=x_padded_length, dtype='complex128')
    x_padded[nperseg:nperseg+x_length] = x

    result = np.zeros(shape=(nfft, x_padded_length), dtype='complex128')

    window = _window(nperseg).astype('complex128')
    window /= np.linalg.norm(window)

    for x_index in range(x_padded_length):
        start_index = x_index - nperseg // 2
        end_index = x_index + nperseg // 2

        segment_temp = np.zeros_like(window, dtype='complex128')

        if end_index < nperseg:
            pass  # All zero, do nothing
        elif start_index > nperseg+x_length:
            pass  # All zero, do nothing
        else:
            segment_temp = x_padded[start_index:end_index]

        result[:, x_index] = fft(segment_temp * window, nfft)

    return result


def _window(length, sigma=1):
    """
    :param length: length of window
    :param sigma: standard deviation
    :return: a gaussian window with length=length
    """

    def gaussian_window(d, s):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-d ** 2 / (2 * s ** 2))

    # x is within 3*sigmas
    x = np.linspace(-3, 3, length)

    window = gaussian_window(d=x, s=sigma)

    return window


def istft(x, nperseg=256, nfft=None):
    """
    :param x: stft result
    :param nperseg: length each segment
    :param nfft: length of fft

    :return: istft result
    """

    if nfft is None:
        nfft = nperseg

    output_shape = x.shape[1]-2*nperseg
    padded_shape = output_shape + 2*nperseg
    window = _window(nperseg).astype('complex128')
    window /= np.linalg.norm(window)

    # deal with margin situation
    result_padded = np.zeros(shape=padded_shape, dtype='complex128')  # length+2*padding

    x = ifft(x, axis=-2)
    # for u in range(padded_shape):
    for u in range(nperseg//2, nperseg+output_shape+nperseg//2):
        result_padded[u - nperseg // 2:u + nperseg // 2] += x[:, u] * window

    # norm square of window
    c = (window*window).sum()
    print(result_padded.shape)

    return result_padded[nperseg:nperseg+output_shape]/c
