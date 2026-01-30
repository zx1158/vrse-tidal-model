# generate_synthetic.py (simplified, safe version)
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--output', type=str, default='synthetic.npz')
    args = parser.parse_args()

    np.random.seed(42)
    n = args.n_samples
    # R = [antiox, protein, temp]
    R = np.random.rand(n, 3)
    # Simulate true sigma (abstract form, no real physics disclosed)
    sigma_true = 0.8 * np.exp(-0.005 * R[:, 1]) / (R[:, 0] + 0.1)
    y = np.random.normal(loc=0, scale=sigma_true)  # dummy target

    np.savez(args.output, R=R, y=y, sigma_true=sigma_true)
    print(f"Saved {n} samples to {args.output}")

if __name__ == "__main__":
    main()
