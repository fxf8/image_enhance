import copy
import fnmatch
import math
import pathlib
import pickle
import random

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.axes
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision.transforms.functional
import tqdm

import image_enhance.database as idb
import image_enhance.model as imodel


class Session:
    models: dict[str, imodel.EnhanceModel]
    model_training_histories: dict[str, imodel.ModelTrainingHistory]
    databases: dict[str, list[idb.ImageSample]]
    name: str | None

    def __init__(self):
        self.models = {}
        self.model_training_histories = {}
        self.databases = {}
        self.name = None

    def __getstate__(self):
        return {
            "models": {
                model_name: model.state_dict()
                for model_name, model in self.models.items()
            },
            "databases": self.databases,
            "name": self.name,
        }

    def __setstate__(self, state):
        self.models = {}

        if "models" in state:
            for model_name, model_state in state["models"].items():
                model: imodel.EnhanceModel = imodel.EnhanceModel()

                model.load_state_dict(model_state)

                self.models[model_name] = model

        self.databases = state["databases"] if "databases" in state else {}
        self.name = state["name"] if "name" in state else None

    def get_name_formatted(self) -> str:
        if self.name is None:
            return "New Unnamed Session"

        return self.name

    def save_session(self, session_path: pathlib.Path):
        with open(session_path, "wb") as session_file:
            self.name = session_path.name

            pickle.dump(self, session_file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_session(session_path: pathlib.Path):
        with open(session_path, "rb") as session_file:
            return pickle.load(session_file)

    def import_glob(self, database_name: str, glob_pattern: str = "*"):
        if database_name in self.databases:
            self.databases[database_name].extend(idb.import_glob(glob_pattern))

        else:
            self.databases[database_name] = idb.import_glob(glob_pattern)

    def corrupt_images(
        self, database_name: str, image_name_glob: str = "*", iterations: int = 1
    ):
        for index in range(len(self.databases[database_name])):
            if fnmatch.fnmatch(
                self.databases[database_name][index].image_path.name, image_name_glob
            ):
                self.databases[database_name][index] = self.databases[database_name][
                    index
                ].corrupt(iterations)

    def merge_databases(self, merged_database_name: str, databases: list[str]):
        self.databases[merged_database_name] = sum(
            (self.databases[database_name] for database_name in databases), []
        )

    def copy_database(self, source_database_name: str, target_database_name: str):
        self.databases[target_database_name] = copy.deepcopy(
            self.databases[source_database_name]
        )

    def delete_database(self, database_name: str):
        del self.databases[database_name]

    def split_database(
        self,
        database_name: str,
        split_ratio: float,
        first_database_name: str,
        second_database_name: str,
    ):
        database_size: int = len(self.databases[database_name])

        self.databases[first_database_name] = self.databases[database_name][
            : int(database_size * split_ratio)
        ]
        self.databases[second_database_name] = self.databases[database_name][
            int(database_size * split_ratio) :
        ]

    def create_model(self, model_name: str):
        self.models[model_name] = imodel.EnhanceModel()
        self.model_training_histories[model_name] = imodel.ModelTrainingHistory([])

    def delete_model(self, model_name: str):
        del self.models[model_name]
        del self.model_training_histories[model_name]

    def get_training_samples(
        self,
        database_name: str,
        sample_expansion: int = 10,
        display_progress: bool = False,
        maximum_dimension: int = 256,
        split_resize_ratio: float = 0.5,
    ) -> list[imodel.ModelTrainingSample]:
        training_samples: list[imodel.ModelTrainingSample] = []

        if display_progress:
            print("Constructing expanded training samples...")

        for sample in (
            tqdm.tqdm(self.databases[database_name])
            if display_progress
            else self.databases[database_name]
        ):
            sample_width, sample_height = sample.get_size()

            if (
                random.random() > split_resize_ratio
                and max(sample_width, sample_height) > maximum_dimension
            ):
                sample_tiles: list[idb.ImageSample] = sample.split_image(
                    (
                        max(1, sample_height // maximum_dimension),
                        max(1, sample_width // maximum_dimension),
                    )
                )

                for tile in sample_tiles:
                    tile_size: tuple[int, int] = tile.get_size()

                    for _ in range(sample_expansion):
                        training_samples.append(
                            imodel.ModelTrainingSample(
                                input=imodel.EnhanceModelInput(
                                    image_sample=tile.corrupt(),
                                    new_size=tile_size,
                                ),
                                expected_output=tile,
                            )
                        )

            else:
                for _ in range(sample_expansion):
                    training_samples.append(
                        imodel.ModelTrainingSample(
                            input=imodel.EnhanceModelInput(
                                image_sample=sample.corrupt(),
                                new_size=sample.get_size(),
                            ),
                            expected_output=sample,
                        )
                    )

        return training_samples

    def train_model(
        self,
        model_name: str,
        database_name: str,
        epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        sample_expansion: int = 10,
        display_progress: bool = False,
        maximum_dimension: int = 256,
        split_resize_ratio: float = 0.5,
    ):
        training_samples: list[imodel.ModelTrainingSample] = self.get_training_samples(
            database_name,
            sample_expansion=sample_expansion,
            display_progress=display_progress,
            maximum_dimension=maximum_dimension,
            split_resize_ratio=split_resize_ratio,
        )

        random.shuffle(training_samples)

        if display_progress:
            print(f"Training model with {len(training_samples)} samples...")

        self.model_training_histories[model_name] = imodel.train_model(
            self.models[model_name],
            training_samples,
            epochs=epochs,
            batch_size=batch_size,
            existing_model_training_history=self.model_training_histories[model_name],
            learning_rate=learning_rate,
            display_progress=display_progress,
        )

    def use_model(
        self,
        model_name: str,
        database_name: str,
        image_glob: str = "*",
        new_size: tuple[int, int] | None = None,
    ) -> bool:
        if model_name not in self.models or database_name not in self.databases:
            return False

        for image_sample in self.databases[database_name]:
            if not fnmatch.fnmatch(image_sample.image_path.name, image_glob):
                continue

            output: torch.Tensor = self.models[model_name](
                imodel.EnhanceModelInput(
                    image_sample=image_sample,
                    new_size=image_sample.get_size() if new_size is None else new_size,
                )
            )

            image_sample.fixed_image_tensor = output

        return True

    def plot_model_training_history(self, model_name: str) -> bool:
        if model_name not in self.model_training_histories:
            return False

        plt.plot(
            [*range(len(self.model_training_histories[model_name].losses))],
            self.model_training_histories[model_name].losses,
        )
        plt.title(f"Model Training History for Model {model_name}")
        plt.xlabel("Batch Number (Iteration Step)")
        plt.ylabel("Loss (L1 Loss)")
        plt.show()

        return True

    def show_images(self, database_name: str, image_glob: str = "*"):
        displayed_image_samples: list[idb.ImageSample] = []

        for image_sample in self.databases[database_name]:
            if fnmatch.fnmatch(image_sample.image_path.name, image_glob):
                image_sample.display_image()
                displayed_image_samples.append(image_sample)

        if len(displayed_image_samples) == 0:
            print("No image samples found.")

            return

        rows_count: int = math.floor(math.sqrt(len(displayed_image_samples)))
        columns_count: int = math.ceil(len(displayed_image_samples) / rows_count)

        figure, axes = plt.subplots(rows_count, columns_count)

        if rows_count == 1 and columns_count == 1:
            axes_grid = np.array([[axes]])
        elif rows_count == 1 or columns_count == 1:
            axes_grid = np.atleast_2d(axes)
        else:
            axes_grid = axes

        for image_sample_index, image_sample in enumerate(displayed_image_samples):
            axis: matplotlib.axes.Axes = axes_grid[
                image_sample_index // columns_count, image_sample_index % columns_count
            ]

            if image_sample.fixed_image_tensor is not None:
                axis_first_half = axis.inset_axes((0, 0, 0.5, 1))
                axis_second_half = axis.inset_axes((0.5, 0, 0.5, 1))

                axis_first_half.imshow(
                    torchvision.transforms.functional.to_pil_image(
                        image_sample.get_tensor()
                    )
                )

                axis_second_half.imshow(
                    torchvision.transforms.functional.to_pil_image(
                        image_sample.fixed_image_tensor
                    )
                )

                axis_first_half.set_title(
                    f"Original Image {image_sample.image_path.name}"
                )
                axis_second_half.set_title(
                    f"Fixed Image {image_sample.image_path.name}"
                )

            else:
                axis.imshow(
                    torchvision.transforms.functional.to_pil_image(
                        image_sample.get_tensor()
                    )
                )
                axis.set_title(f"Original Image {image_sample.image_path.name}")

            axis.axis("off")

        figure.tight_layout()

        plt.show()
