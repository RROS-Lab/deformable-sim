import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="result/package_parameter_sampling",
        help="Directory containing the result files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["evaluation", "benchmark"],
        default="evaluation",
        help="Mode to plot",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    mode = args.mode

    if not os.path.exists(os.path.join(os.getcwd(), input_dir)):
        raise FileNotFoundError(f"Directory {input_dir} does not exist")
    if not os.path.exists(os.path.join(os.getcwd(), input_dir, mode)):
        raise FileNotFoundError(f"Directory {mode} does not exist")
    if not os.path.exists(os.path.join(os.getcwd(), input_dir, mode, f"{mode}.csv")):
        raise FileNotFoundError(f"File {mode}.csv does not exist")

    df = pd.read_csv(os.path.join(os.getcwd(), input_dir, mode, f"{mode}.csv"))

    for idx, row in df.iterrows():
        total_losses = (
            row["total_losses"][1:-1]
            .replace("np.float32(", "")
            .replace(")", "")
            .split(", ")
        )
        package_particle_losses = (
            row["package_particle_losses"][1:-1]
            .replace("np.float64(", "")
            .replace(")", "")
            .split(", ")
        )
        internal_object_losses = (
            row["internal_object_losses"][1:-1]
            .replace("np.float64(", "")
            .replace(")", "")
            .split(", ")
        )

        total_losses = [float(i) for i in total_losses]
        package_particle_losses = [float(i) for i in package_particle_losses]
        internal_object_losses = [float(i) for i in internal_object_losses]

        print(
            "Iteration",
            row["iteration"] if mode == "evaluation" else row["particle_in_between"],
        )
        print("Total Loss Mean", np.mean(total_losses))
        print("Total Loss Std", np.std(total_losses))
        print("Package Particle Loss Mean", np.mean(package_particle_losses))
        print("Package Particle Loss Std", np.std(package_particle_losses))
        print("Internal Object Loss Mean", np.mean(internal_object_losses))
        print("Internal Object Loss Std", np.std(internal_object_losses))


if __name__ == "__main__":
    main()
