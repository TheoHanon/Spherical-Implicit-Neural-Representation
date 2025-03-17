import matplotlib.pyplot as plt
import numpy as np

def plot_SHT_coeffs(coeffs, fig=None, ticks_m=10, ticks_l=10, title=None,
                    colorbar=False, **kwargs):
    
    
    Lmax = len(coeffs) - 1

    # Compute the log-magnitude, handling zeros.
    abs_coeffs = np.abs(coeffs)
    abs_coeffs[abs_coeffs == 0] = np.nan  # prevent log(0)
    log_coeffs = np.log(abs_coeffs)

    
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = fig.add_subplot(111)

    # Define edges so that each coefficient is centered.
    m_edges = np.linspace(-Lmax - 0.5, Lmax + 0.5, coeffs.shape[1] + 1)
    l_edges = np.linspace(-0.5, Lmax + 0.5, coeffs.shape[0] + 1)

    # Plot the data.
    im = ax.imshow(log_coeffs, origin='lower',
                   extent=[m_edges[0], m_edges[-1], l_edges[0], l_edges[-1]],
                   aspect='equal', **kwargs)

    # Set tick marks.
    m_ticks = np.arange(-Lmax, Lmax + 1, ticks_m)
    ax.set_xticks(m_ticks)
    ax.set_xticklabels(m_ticks)
    l_ticks = np.arange(0, Lmax + 1, ticks_l)
    ax.set_yticks(l_ticks)
    ax.set_yticklabels(l_ticks)

    ax.set_xlabel("m", fontsize=16)
    ax.set_ylabel("ℓ", fontsize=16)
    # Place the m-axis on the top.
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Hide spines for a cleaner look.
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if title:
        ax.set_title(title, y=1.05)

    # Optionally add a horizontal colorbar.
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, aspect=20, pad=0.05, orientation='horizontal')
        cbar.set_label("Log |Coefficient|", fontsize=10)

    plt.tight_layout()
    return im




def sample_s2(L: int, sampling: str = "gl"):
    """
    Samples points on the 2-sphere for a given resolution L and sampling type.

    Parameters:
    L (int): Bandwidth of the spherical harmonics.
    sampling (str): Sampling scheme, default is 'gl' (Gauss-Legendre).

    Returns:
    tuple: A tuple containing:
        - phi (numpy.ndarray): Longitudinal angles (azimuth).
        - theta (numpy.ndarray): Latitudinal angles (colatitude).
        - (nlon, nlat) (tuple): Number of longitude and latitude points.
    """

    import torch
    from s2fft.sampling.s2_samples import phis_equiang, thetas

    phi = phis_equiang(L, sampling=sampling)
    theta = thetas(L, sampling=sampling)
    nlon, nlat = phi.shape[0], theta.shape[0]

    phi, theta = np.meshgrid(phi, theta)
    phi, theta = torch.tensor(phi), torch.tensor(theta)

    return phi, theta, (nlon, nlat)
