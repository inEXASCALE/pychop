import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def set_publication_style(fontsize=12):
    """
    Configure matplotlib for clean publication-quality figures.
    """

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "figure.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "legend.frameon": False
    })


def visualize(csv_path='results/pychop_overhead.csv', fontsize=12):

    set_publication_style(fontsize)

    backend_name_map = {
        "numpy": "NumPy",
        "jax": "JAX",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "cupy": "CuPy",
        "mkl": "MKL",
        "openblas": "OpenBLAS",
        "oneapi": "oneAPI"
    }

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)

    for n in [2000, 5000]:

        df_n = df[df['size'] == n]
        raw_backends = df_n['backend'].unique()

        natives = []
        emulateds = []
        display_names = []

        for b in raw_backends:
            native_time = df_n[
                (df_n['backend'] == b) &
                (df_n['mode'] == 'native')
            ]['time'].values[0]

            emu_time = df_n[
                (df_n['backend'] == b) &
                (df_n['mode'] == 'emulated')
            ]['time'].values[0]

            natives.append(native_time)
            emulateds.append(emu_time)
            display_names.append(
                backend_name_map.get(b.lower(), b)
            )

        # Sort by native time for cleaner structure
        combined = list(zip(display_names, natives, emulateds))
        combined.sort(key=lambda x: x[1])
        display_names, natives, emulateds = zip(*combined)

        x = np.arange(len(display_names))
        width = 0.36

        # Wider figure to avoid crowding
        fig, ax = plt.subplots(figsize=(6.5, 4.8))

        native_color = "#4C72B0"
        emu_color = "#DD8452"

        ax.bar(x - width/2, natives, width,
               label="Native FP32",
               color=native_color,
               edgecolor="black",
               linewidth=0.7)

        ax.bar(x + width/2, emulateds, width,
               label="Emulated FP32 in Pychop",
               color=emu_color,
               edgecolor="black",
               linewidth=0.7)

        ax.set_xlabel("Backend")
        ax.set_ylabel("Execution Time (seconds)")

        ax.set_xticks(x)
        ax.set_xticklabels(display_names)

        ax.grid(axis='y')
        ax.set_axisbelow(True)

        ax.legend()

        plt.tight_layout()

        fig_name = f"overhead_comparison_size_{n}"

        plt.savefig(output_dir / f"{fig_name}.png",
                    dpi=600,
                    bbox_inches='tight')
        plt.savefig(output_dir / f"{fig_name}.pdf",
                    bbox_inches='tight')
        plt.savefig(output_dir / f"{fig_name}.svg",
                    bbox_inches='tight')

        plt.close(fig)

    print("Clean publication-quality figures saved in 'figures/' directory.")


if __name__ == '__main__':
    visualize()