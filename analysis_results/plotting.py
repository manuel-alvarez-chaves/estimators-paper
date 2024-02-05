import matplotlib.pyplot as plt

cm = 1 / 2.54  # inch to centimeters


def small_grid():
    sample_size_range = [60, 120_000]

    fig = plt.figure(figsize=(15.5 * cm, 10 * cm))
    b = 0.295
    h = 0.430
    ax1 = fig.add_axes([0.05, 0.525, b, h])
    ax2 = fig.add_axes([0.05, 0.08, b, h])
    ax3 = fig.add_axes([0.3525, 0.525, b, h])
    ax4 = fig.add_axes([0.3525, 0.08, b, h])
    ax5 = fig.add_axes([0.655, 0.525, b, h])
    ax6 = fig.add_axes([0.655, 0.08, b, h])
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    ax1.set_title("KDE estimator")
    ax3.set_title("binning estimator")
    ax5.set_title("$k$-NN estimator")
    ax1.set_ylabel("Entropy (nats)")
    ax2.set_ylabel("$D_{KL}$ (nats)")
    ax4.set_xlabel("sample size")

    for ax in axes:
        for label in ax.get_yticklabels():
            label.set_rotation(90)
            label.set_va("center")
        ax.set_xlim(sample_size_range)
        ax.set_xscale("log")

    for ax in [ax3, ax4, ax5, ax6]:
        ax.yaxis.set_ticks_position("none")
        ax.set_yticklabels([])

    for ax in [ax1, ax3, ax5]:
        ax.xaxis.set_ticks_position("none")
        ax.set_xticklabels([])

    return fig, axes
