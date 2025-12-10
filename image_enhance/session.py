import pathlib
import pickle
import random

import tqdm

import image_enhance.database as idb
import image_enhance.model as imodel


class Session:
    models: dict[str, imodel.EnhanceModel]
    databases: dict[str, list[idb.ImageSample]]

    def __init__(self):
        self.models = {}
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
        self.databases[database_name] = idb.import_glob(glob_pattern)

    def merge_databases(self, merged_database_name: str, databases: list[str]):
        self.databases[merged_database_name] = sum(
            (self.databases[database_name] for database_name in databases), []
        )

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
