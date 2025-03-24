import os
import pandas as pd
import numpy as np
from loguru import logger
import yaml
import argparse
from dataclasses import dataclass
import pickle
from src.data_preparer.containers import Marker, Package, Tool, InsideObject
from src.data_preparer.utils import (
    process_timestamped_trajectory,
    process_marker_timestamped_trajectory,
    process_marker_timestamped_trajectory_no_id,
    visualize_no_id_package_motion,
    visualize_markers,
    array_to_SE3,
    SE3_to_array,
    find_missing_centroid,
)

# Appending paths as a temporary
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


@dataclass
class ConfigFileKeys:
    dataset_dir = "dataset_dir"
    skip_rows = "skip_rows"
    verbose = "verbose"
    parsing_order = "ParsingOrder"
    package_size = "PackageSizes"
    object_sizes = "ObjectSizes"
    package_masses = "PackageMasses"
    object_masses = "ObjectMasses"
    weights = "Weights"
    package_num_markers = "PackageNumMarkers"
    object_num_markers = "ObjectNumMarkers"
    packages_to_process = "PackagesToProcess"
    object_size_to_process = "ObjectSizeToProcess"
    object_mass_to_process = "ObjectMassToProcess"
    include_failures = "IncludeFailures"
    world_T_tool = "world_T_tool"
    world_T_cup = "world_T_cup"


@dataclass
class MetaDataKeys:
    package_size = "Package Size"
    object_size = "Object Size"
    object_mass = "Object Mass"
    velocity = "Velocity"
    final_orientation = "Final Orientation"
    filename = "Filename"
    failure = "Failure"


class Parser:

    def __init__(self, parser_config: dict):
        self.parser_config = parser_config
        self.dataset_dir = parser_config["dataset_dir"]
        self.verbose = parser_config["verbose"]
        self.metadata_file = parser_config["metadata_file"]
        ##Reading Data
        self.skip_rows = self.parser_config["skip_rows"]

        self._load_data()

        return

    def _load_data(self):

        # Loading the metadata
        logger.info("Loading the dataset...")
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset directory {self.dataset_dir} does not exist."

        self.metadata_df = pd.read_csv(self.metadata_file, index_col=False)

        list_of_csv_files = [
            entry.name
            for entry in os.scandir(self.dataset_dir)
            if entry.is_file() and entry.name.endswith(".csv")
        ]

        logger.info(
            f"Found {len(list_of_csv_files)} CSV files in the dataset directory."
        )
        total_files_processed = 0
        for filecount, csv_file in enumerate(list_of_csv_files):
            current_csv_file = os.path.join(self.dataset_dir, csv_file)
            data_filename = csv_file.split(os.sep)[-1].split(".")[0]
            logger.info(f"Processing {csv_file}...")
            if not self.parser_config[ConfigFileKeys.include_failures]:
                if self.metadata_df[MetaDataKeys.failure][
                    np.where(self.metadata_df[MetaDataKeys.filename] == data_filename)[
                        0
                    ][0]
                ].astype(bool):
                    logger.info(f"Skipping {csv_file} as it is a failure case.")
                    continue
            if (
                data_filename[
                    self.parser_config[ConfigFileKeys.parsing_order]["package_size"]
                ]
                not in self.parser_config[ConfigFileKeys.packages_to_process]
            ):
                logger.info(
                    f"Skipping {csv_file} as package size is not in the list of packages to process."
                )
                continue
            if (
                data_filename[
                    self.parser_config[ConfigFileKeys.parsing_order]["object_size"]
                ]
                not in self.parser_config[ConfigFileKeys.object_size_to_process]
            ):
                logger.info(
                    f"Skipping {csv_file} as object size is not in the list of object sizes to process."
                )
                continue
            if (
                data_filename[
                    self.parser_config[ConfigFileKeys.parsing_order]["object_mass"]
                ]
                not in self.parser_config[ConfigFileKeys.object_mass_to_process]
            ):
                logger.info(
                    f"Skipping {csv_file} as object mass is not in the list of object masses to process."
                )
                continue

            ##Ensuring that Ultra High Velocity is Not Porcessed
            if (
                data_filename[
                    self.parser_config[ConfigFileKeys.parsing_order]["velocity"]
                ]
                == "U"
            ):
                logger.info(
                    f"Skipping {csv_file} as object mass is not in the list of object masses to process."
                )
                continue

            self._process_csv_file(current_csv_file)
            logger.info(f"Processed {csv_file}.")
            total_files_processed += 1

        if self.verbose:
            logger.info(f"Total files processed: {total_files_processed}")
            logger.info("Dataset loaded successfully.")
        return

    def _process_csv_file(self, csv_file):

        assert os.path.exists(csv_file), f"File {csv_file} does not exist."
        mocap_dataframe = pd.read_csv(
            csv_file, skiprows=np.arange(self.skip_rows), index_col=False
        )
        marker_information = pd.read_csv(
            csv_file,
            skiprows=lambda x: x not in [3],
            index_col=False,
        )
        marker_information = np.array(marker_information.columns.to_list()[2:])
        unlabeled_marker_ids = (
            np.where(np.char.startswith(marker_information, "Unlabeled"))[0]
            + 2  # NOTE: 2 is added to compensate for first 2 columsns of the csv file
        )

        data_filename = csv_file.split(os.sep)[-1].split(".")[0]
        assert len(data_filename) == 5, f"Filename {data_filename} is not valid."

        package_size_key = data_filename[
            self.parser_config[ConfigFileKeys.parsing_order]["package_size"]
        ]
        package_dims = self.parser_config[ConfigFileKeys.package_size][package_size_key]
        object_size_key = data_filename[
            self.parser_config[ConfigFileKeys.parsing_order]["object_size"]
        ]
        object_size = self.parser_config[ConfigFileKeys.object_sizes][object_size_key]
        package_mass = self.parser_config[ConfigFileKeys.package_masses][
            package_size_key
        ]
        object_mass_key = data_filename[
            self.parser_config[ConfigFileKeys.parsing_order]["object_mass"]
        ]

        object_mass = self.parser_config[ConfigFileKeys.object_masses][object_mass_key]
        weights = (
            self.parser_config[ConfigFileKeys.weights][object_mass_key] * 0.453592
        )  # Converting lbs to kg

        total_object_mass = object_mass + weights

        total_mass = package_mass + total_object_mass

        if self.verbose:
            logger.info(f"Total mass: {total_mass} kg")
            logger.info(f"Package size: {package_dims} m")
            logger.info(f"Object size: {object_size} m")

        # Defining the Tool
        tool: Tool = self.process_tool(mocap_dataframe)
        # save_path_dir = os.path.join(
        #     MAIN_DIR, "src", "data_preparer", "plots", "tool_trajectory"
        # )
        # if not os.path.exists(save_path_dir):
        #     os.makedirs(save_path_dir)
        # save_path = os.path.join(save_path_dir, f"{data_filename}_tool_trajectory.png")
        # tool.visualize_trajectory_2D(save_path=save_path)
        ###Defining the package
        # package: Package = self.process_package(
        #     mocap_dataframe,
        #     package_dims,
        #     package_type=package_size_key,
        #     object_type=object_size_key,
        #     start_of_motion=tool.start_of_motion,
        #     end_of_motion=tool.end_of_motion,
        #     unlabeled_marker_ids=unlabeled_marker_ids,
        #     datafilename=None,
        # )

        initial_marker_positions, package_timestamped_trajectory = (
            self.process_package_no_id(
                mocap_dataframe,
                package_dims,
                package_type=package_size_key,
                object_type=object_size_key,
                start_of_motion=tool.start_of_motion,
                end_of_motion=tool.end_of_motion,
                unlabeled_marker_ids=unlabeled_marker_ids,
                datafilename=data_filename,
            )
        )
        last_key = list(package_timestamped_trajectory.keys())[-1]  # Get the last key
        package_timestamped_trajectory.pop(last_key)  # Remove it

        package_info = {}
        package_info["total_mass"] = total_mass
        package_info["package_mass"] = package_mass
        package_info["package_dims"] = package_dims
        package_info["object_mass"] = object_mass
        ##Defining Inside Object Params
        inside_object = self.process_object(
            mocap_dataframe=mocap_dataframe,
            object_size=object_size,
            object_mass=total_object_mass,
            start_of_motion=tool.start_of_motion,
            end_of_motion=tool.end_of_motion,
            marker_trajectory=tool.tool_positions,
        )

        initial_tool_position = np.hstack(
            (tool.tool_positions[0], tool.tool_orientations[0])
        )

        tool.filter_tool_trajectory_on_start_end_time()
        inside_object.filter_object_trajectory_on_start_end_time()

        initial_suction_cup_position = self.get_initial_suction_cup_position(
            initial_tool_position, initial_marker_positions
        )

        self.save_processed_data(
            datafilename=data_filename,
            package=package_timestamped_trajectory,
            inside_object=inside_object,
            tool=tool,
            package_info=package_info,
            initial_marker_positions=initial_marker_positions,
            initial_suction_cup_position=initial_suction_cup_position,
        )

        # visualize_no_id_package_motion(package=package_timestamped_trajectory, tool=tool, object=inside_object)
        # ##Plotting the package at start of motion
        # viz_timestamp = tool.start_of_motion
        # # viz_timestamp = 0.0
        # start_of_motion_idx = np.argmin(
        #     np.abs(tool.timestamps - viz_timestamp)
        # )  # Finding the index of the start of motion
        # tool_tf_list = []
        # inside_object_tf_list = []
        # for t in tool.timestamps[start_of_motion_idx:]:
        #     tool_tf_list.append(
        #         np.hstack(
        #             (
        #                 tool.tool_positions[np.argmin(np.abs(tool.timestamps - t))],
        #                 tool.tool_orientations[np.argmin(np.abs(tool.timestamps - t))],
        #             )
        #         )
        #     )
        #     inside_object_tf_list.append(
        #         np.hstack(
        #             (
        #                 inside_object.object_positions[
        #                     np.argmin(np.abs(tool.timestamps - t))
        #                 ],
        #                 np.array([0, 0, 0, 1]),
        #             )
        #         )
        #     )

        # for i in range(start_of_motion_idx, len(tool.timestamps)):
        #     # if i >= len(tool.timestamps) - 10:
        #     package.visualize_package_at_timestamp(
        #         timestamp=tool.timestamps[i],
        #         all_timestamps=tool.timestamps,
        #         suction_tool_tf=tool_tf_list[i],
        #         object_tf=inside_object_tf_list[i],
        #         world_tf=np.array([0, 0, 0, 0, 0, 0, 1]),
        #     )

        # package.visualize_package_motion(
        #     start_idx=start_of_motion_idx,
        #     timestamps=tool.timestamps[start_of_motion_idx:],
        #     suction_tool_tf_list=tool_tf_list,
        #     object_tf_list=inside_object_tf_list,
        #     saving_filename=f"{data_filename}_motion.gif",
        # )

        return

    def get_initial_suction_cup_position(
        self, initial_tool_position, initial_marker_positions, visualize: bool = True
    ):

        world_T_tool = self.parser_config[ConfigFileKeys.world_T_tool]
        world_T_cup = self.parser_config[ConfigFileKeys.world_T_cup]

        world_T_tool_mat = array_to_SE3(world_T_tool)
        world_T_cup_mat = array_to_SE3(world_T_cup)

        tool_T_cup_mat = np.matmul(np.linalg.inv(world_T_tool_mat), world_T_cup_mat)
        world_T_tool_mat_curr = array_to_SE3(initial_tool_position)
        world_T_cup_mat_curr = np.matmul(world_T_tool_mat_curr, tool_T_cup_mat)
        world_T_cup_updated = SE3_to_array(world_T_cup_mat_curr)
        # centroid = find_missing_centroid(marker_positions=initial_marker_positions)
        # world_T_cup_updated[:3] = centroid
        if visualize:
            frames = [
                ["Tool", initial_tool_position],
                ["cup", world_T_cup_updated],
                ["world", np.array([0, 0, 0, 0, 0, 0, 1])],
            ]
            visualize_markers(initial_marker_positions, frames)

        return world_T_cup_updated[:3]

    def save_processed_data(
        self,
        datafilename,
        package: dict,
        inside_object: InsideObject,
        tool: Tool,
        package_info: dict,
        initial_marker_positions: np.ndarray,
        initial_suction_cup_position: np.ndarray,
    ):
        # Saving the processed data
        save_path_dir = os.path.join(MAIN_DIR, "src", "data_preparer", "processed_data")
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        save_path = os.path.join(save_path_dir, f"{datafilename}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(
                {
                    "timestamp": tool.timestamps,
                    "marker_trajectories": package,
                    "tool_positions": tool.tool_positions,
                    "tool_orientations": tool.tool_orientations,
                    "tool_velocity": tool.tool_velocity,
                    "tool_acceleration": tool.tool_acceleration,
                    "object_positions": inside_object.object_positions,
                    "object_velocities": inside_object.object_velocity,
                    "object_acceleration": inside_object.object_acceleration,
                    "object_mass": inside_object.object_mass,
                    "package_info": package_info,
                    "initial_suction_cup_position": initial_suction_cup_position,
                    "initial_marker_positions": initial_marker_positions,
                },
                f,
            )

    def process_tool(self, mocap_dataframe: pd.DataFrame):
        tool = Tool(tool_name="four_suction_cup_tool")

        mocap_dataframe_keys = mocap_dataframe.keys()
        timestamps = mocap_dataframe[mocap_dataframe_keys[1]].to_numpy()
        tool_positions, tool_orientations = process_timestamped_trajectory(
            mocap_dataframe,
            xyz_start_end_ids=[6, 9],
            orientation_start_end_ids=[2, 6],
            verbose=self.verbose,
        )

        tool.initialize_tool_trajectory(
            timestamps=timestamps,
            tool_positions=tool_positions,
            tool_orientations=tool_orientations,
        )

        tool.compute_tool_velocity()
        ##Computing Start of Motion
        start_of_motion = timestamps[
            np.where(np.abs(tool.tool_velocity[:, 2]) > 0.01)[0][0]
        ]

        # Find the first occurrence of a long enough flat region at the end
        end_of_motion = timestamps[
            np.where(np.abs(tool.tool_velocity[:, 2]) > 0.01)[0][-1]
        ]

        tool.start_of_motion = start_of_motion
        tool.end_of_motion = end_of_motion

        logger.info(f"Start of Motion: {start_of_motion}")
        logger.info(f"End of Motion: {end_of_motion}")
        tool.compute_tool_acceleration()

        ##NOTE(OMEY): Add Suction Cup Deformation Computation Later On

        if self.verbose:
            logger.info("Tool trajectory initialized.")
            logger.info("Tool velocity computed.")
            logger.info("Tool acceleration computed.")

        return tool

    def process_markers(self, mocap_dataframe: pd.DataFrame, filename: str):

        # Computing the end of Motion
        ##Check if Failure case
        failure = self.metadata_df[MetaDataKeys.failure][
            np.where(self.metadata_df[MetaDataKeys.filename] == filename)[0][0]
        ].astype(bool)

        if self.verbose:
            logger.info(f"Failure: {failure}")

        return

    def process_package_no_id(
        self,
        mocap_dataframe: pd.DataFrame,
        package_dims: list,
        package_type: str,
        object_type: str,
        start_of_motion: float,
        end_of_motion: float,
        unlabeled_marker_ids: np.ndarray,
        datafilename: str = None,
    ):
        package = {}
        initial_marker_positions = []
        mocap_dataframe_keys = mocap_dataframe.keys()
        num_markers = self.parser_config[ConfigFileKeys.package_num_markers][
            package_type
        ]
        object_num_markers = self.parser_config[ConfigFileKeys.object_num_markers][
            object_type
        ]

        if self.verbose:
            logger.info(f"Number of markers on the object: {object_num_markers}")

        marker_data_start_idx = unlabeled_marker_ids[0]
        marker_data_end_idx = unlabeled_marker_ids[-3]

        ##Checking how many markers are registered in the csv
        if self.verbose:
            number_of_package_markers = (
                len(mocap_dataframe_keys[marker_data_start_idx:marker_data_end_idx])
                // 3
            )
            logger.info(
                f"Number of markers registered in the csv: {number_of_package_markers}"
            )
            logger.info(f"Number of markers expected: {num_markers}")

        saving_fileheader = None
        if datafilename is not None:
            plot_saving_dir = os.path.join(
                MAIN_DIR, "src", "data_preparer", "plots", "marker_trajectories"
            )

            if not os.path.exists(plot_saving_dir):
                os.makedirs(plot_saving_dir)
            saving_fileheader = os.path.join(plot_saving_dir, datafilename)

        initial_marker_position, package = process_marker_timestamped_trajectory_no_id(
            mocap_dataframe,
            package_dims=package_dims,
            xyz_start_end_ids=[marker_data_start_idx, marker_data_end_idx],
            num_markers=num_markers,
            start_of_motion=start_of_motion,
            end_of_motion=end_of_motion,
            saving_fileheader=saving_fileheader,
        )

        return initial_marker_position, package

    def process_package(
        self,
        mocap_dataframe: pd.DataFrame,
        package_dims: list,
        package_type: str,
        object_type: str,
        start_of_motion: float,
        end_of_motion: float,
        unlabeled_marker_ids: np.ndarray,
        datafilename: str = None,
    ):
        package = Package(package_dims=package_dims, package_type=package_type)
        num_markers = self.parser_config[ConfigFileKeys.package_num_markers][
            package_type
        ]
        package.start_of_motion = start_of_motion
        package.end_of_motion = end_of_motion

        mocap_dataframe_keys = mocap_dataframe.keys()
        timestamps = mocap_dataframe[mocap_dataframe_keys[1]].to_numpy()
        object_num_markers = self.parser_config[ConfigFileKeys.object_num_markers][
            object_type
        ]

        if self.verbose:
            logger.info(f"Number of markers on the object: {object_num_markers}")

        marker_data_start_idx = unlabeled_marker_ids[0]
        marker_data_end_idx = unlabeled_marker_ids[-3]

        ##Checking how many markers are registered in the csv
        if self.verbose:
            number_of_package_markers = (
                len(mocap_dataframe_keys[marker_data_start_idx:marker_data_end_idx])
                // 3
            )
            logger.info(
                f"Number of markers registered in the csv: {number_of_package_markers}"
            )
            logger.info(f"Number of markers expected: {num_markers}")

        saving_fileheader = None
        if datafilename is not None:
            plot_saving_dir = os.path.join(
                MAIN_DIR, "src", "data_preparer", "plots", "marker_trajectories"
            )

            if not os.path.exists(plot_saving_dir):
                os.makedirs(plot_saving_dir)
            saving_fileheader = os.path.join(plot_saving_dir, datafilename)

        marker_trajectories: dict = process_marker_timestamped_trajectory(
            mocap_dataframe,
            xyz_start_end_ids=[marker_data_start_idx, marker_data_end_idx],
            num_markers=num_markers,
            plot=False,
            saving_fileheader=saving_fileheader,
        )

        for marker_id, marker_trajectory in marker_trajectories.items():

            current_marker = Marker(
                marker_id=marker_id,
                package_id=package_type,
            )
            current_marker.set_timestamped_trajectory(
                timestamps=marker_trajectory[:, 0],
                timestamped_trajectory=marker_trajectory[:, 1:],
            )
            current_marker.start_of_motion = start_of_motion
            current_marker.end_of_motion = end_of_motion
            package.add_marker(current_marker)

        return package

    def process_object(
        self,
        mocap_dataframe,
        object_size,
        object_mass,
        start_of_motion,
        end_of_motion,
        marker_trajectory=None,
    ):
        inside_object = InsideObject(object_size=object_size, object_mass=object_mass)
        inside_object.start_of_motion = start_of_motion
        inside_object.end_of_motion = end_of_motion

        mocap_dataframe_keys = mocap_dataframe.keys()
        timestamps = mocap_dataframe[mocap_dataframe_keys[1]].to_numpy()
        object_orientation_w_idx = np.where(mocap_dataframe_keys == "W.1")[0][0]
        start_idx = object_orientation_w_idx + 1
        object_positions, _ = process_timestamped_trajectory(
            mocap_dataframe,
            xyz_start_end_ids=[
                start_idx,
                start_idx + 3,
            ],
            orientation_start_end_ids=None,
            verbose=self.verbose,
            object=True,
            marker_trajectory=marker_trajectory,
        )

        inside_object.initialize_object_trajectory(
            timestamps=timestamps, object_positions=object_positions
        )

        inside_object.compute_object_velocity()
        inside_object.compute_object_acceleration()
        # inside_object.visualize_velocities()

        if self.verbose:
            logger.info("Object trajectory initialized.")
            logger.info("Object velocity computed.")
            logger.info("Object acceleration computed.")

        return inside_object

    def visualize():

        return


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Parser for the dataset")
    args.add_argument(
        "-f",
        "--config_file",
        type=str,
        default="data_config.yaml",
        help="Dataset directory",
    )
    args = args.parse_args()

    config_file = os.path.join(
        MAIN_DIR, "src", "data_preparer", "config", args.config_file
    )
    with open(config_file, "r") as file:
        parser_config = yaml.safe_load(file)

    parser_config["dataset_dir"] = os.path.join(MAIN_DIR, parser_config["dataset_dir"])
    parser_config["metadata_file"] = os.path.join(
        MAIN_DIR, parser_config["metadata_file"]
    )
    parser = Parser(parser_config=parser_config)
