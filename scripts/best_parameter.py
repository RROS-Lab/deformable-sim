import argparse
import os

import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, default="result/package_parameter_sampling"
    )
    parser.add_argument("--parameter", type=str, default="package_parameter.csv")
    parser.add_argument("--index", type=str, default="train_test_index.csv")
    parser.add_argument("--output", type=str, default="best.csv")
    args = parser.parse_args()

    input_dir = args.input_dir
    parameter_file = os.path.join(input_dir, args.parameter)
    index_file = os.path.join(input_dir, args.index)
    if not os.path.exists(input_dir):
        FileNotFoundError(f"Directory {input_dir} does not exist.")
        return
    if not os.path.exists(parameter_file):
        FileNotFoundError(f"File {parameter_file} does not exist.")
        return
    if not os.path.exists(index_file):
        FileNotFoundError(f"File {index_file} does not exist.")
        return

    index_df = pd.read_csv(index_file)

    train_index = str(index_df["train_index"].array[0])
    test_index = str(index_df["test_index"].array[0])

    parameter_df = pd.read_csv(parameter_file)
    output_df = pd.DataFrame(
        columns=[
            "iteration",
            "package_tri_ke",
            "package_tri_ka",
            "package_tri_kd",
            "package_edge_ke",
            "package_edge_kd",
            "spring_ke",
            "spring_kd",
            "shape_ke",
            "shape_kd",
            "shape_kf",
            "shape_mu",
            "train_index",
            "test_index",
            "internal_object_loss",
            "package_particle_loss",
            "total_loss",
        ]
    )

    minimum_mean_loss = float("inf")

    for index, row in parameter_df.iterrows():
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

        mean_loss = np.mean(total_losses)

        if mean_loss < minimum_mean_loss:
            minimum_mean_loss = mean_loss

            row_df = pd.DataFrame(
                [
                    row["iteration"],
                    row["package_tri_ke"],
                    row["package_tri_ka"],
                    row["package_tri_kd"],
                    row["package_edge_ke"],
                    row["package_edge_kd"],
                    row["spring_ke"],
                    row["spring_kd"],
                    row["shape_ke"],
                    row["shape_kd"],
                    row["shape_kf"],
                    row["shape_mu"],
                    train_index,
                    test_index,
                    np.mean(internal_object_losses),
                    np.mean(package_particle_losses),
                    mean_loss,
                ],
                index=output_df.columns,
            )
            output_df = pd.concat([output_df, row_df.T], ignore_index=True)

    output_df.to_csv(os.path.join(input_dir, args.output))


if __name__ == "__main__":
    main()
