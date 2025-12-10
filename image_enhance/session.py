import copy
import fnmatch
import pathlib
import pickle
import random

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

import torch
import tqdm

import image_enhance.database as idb
import image_enhance.model as imodel


class Session:
    models: dict[str, imodel.EnhanceModel]
    model_training_histories: dict[str, imodel.ModelTrainingHistory]
    databases: dict[str, list[idb.ImageSample]]

    def __init__(self):
        self.models = {}
        self.model_training_histories = {}
        self.databases = {}

    def __getstate__(self):
        return {
            "models": {
                model_name: model.state_dict()
                for model_name, model in self.models.items()
            },
            "databases": self.databases,
        }

    def __setstate__(self, state):
        self.models = {}

        for model_name, model_state in state["models"].items():
            model: imodel.EnhanceModel = imodel.EnhanceModel()

            model.load_state_dict(model_state)

            self.models[model_name] = model

        self.databases = state["databases"]

    def save_session(self, session_path: pathlib.Path):
        with open(session_path, "wb") as session_file:
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
