import pickle

import image_enhance.database as idb
import image_enhance.model as imodel


class Session:
    models: dict[str, imodel.EnhanceModel]
    databases: dict[str, list[idb.ImageSample]]

    def __init__(self):
        self.model = {}
        self.database = {}

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

    def train_model(
        self,
        model_name: str,
        database_name: str,
        batch_size: int = 32,
        epochs: int = 1,
        learning_rate=1e-3,
    ): ...
