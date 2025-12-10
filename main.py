import image_enhance.interface as ui
import image_enhance.actions as iactions


def main():
    DIOLAGUE_TREE: list[ui.MenuOption] = [
        ui.MenuOption(
            name="Manage Session",
            header="--- Session Management ---",
            suboptions=[
                ui.MenuOption(
                    name="Save Session",
                    description="Save the current session state to a file.",
                    callback=iactions.save_session,
                ),
                ui.MenuOption(
                    name="Load Session",
                    description="Load a session state from a file.",
                    callback=iactions.load_session,
                ),
            ],
        ),
        ui.MenuOption(
            name="Manage Databases",
            header="--- Database Management ---",
            suboptions=[
                ui.MenuOption(
                    name="List Databases",
                    description="List all available image databases in the session.",
                    callback=iactions.list_databases,
                ),
                ui.MenuOption(
                    name="Create Empty Dataset",
                    description="Create a new, empty dataset.",
                    callback=iactions.create_dataset,
                ),
                ui.MenuOption(
                    name="Import Images from Glob",
                    description="Create or add to a database from a glob pattern.",
                    callback=iactions.import_from_glob,
                ),
                ui.MenuOption(
                    name="Show Images",
                    description="Display images from a database.",
                    callback=iactions.show_images_in_database,
                ),
                ui.MenuOption(
                    name="Export Images",
                    description="Export original or fixed images from a database to a directory.",
                    callback=iactions.export_images,
                ),
                ui.MenuOption(
                    name="Copy Database",
                    description="Create a copy of an existing database.",
                    callback=iactions.copy_database,
                ),
                ui.MenuOption(
                    name="Split Database",
                    description="Split a database into two new ones.",
                    callback=iactions.split_database,
                ),
                ui.MenuOption(
                    name="Merge Databases",
                    description="Merge multiple databases into a new one.",
                    callback=iactions.merge_databases,
                ),
                ui.MenuOption(
                    name="Corrupt Database",
                    description="Apply random corruptions to images in a database.",
                    callback=iactions.corrupt_database,
                ),
                ui.MenuOption(
                    name="Delete Database",
                    description="Delete a database from the session.",
                    callback=iactions.delete_database,
                ),
            ],
        ),
        ui.MenuOption(
            name="Manage Models",
            header="--- Model Management ---",
            suboptions=[
                ui.MenuOption(
                    name="List Models",
                    description="List all available models in the session.",
                    callback=iactions.list_models,
                ),
                ui.MenuOption(
                    name="Create New Model",
                    description="Create a new, untrained enhancement model.",
                    callback=iactions.create_model,
                ),
                ui.MenuOption(
                    name="Train Model",
                    description="Train a model on a database of images.",
                    callback=iactions.train_model,
                ),
                ui.MenuOption(
                    name="Use Model",
                    description="Apply a model to a database of images.",
                    callback=iactions.use_model,
                ),
                ui.MenuOption(
                    name="Plot Model Training History",
                    description="Plot the loss history for a trained model.",
                    callback=iactions.plot_model_training_history,
                ),
                ui.MenuOption(
                    name="Delete Model",
                    description="Delete a model from the session.",
                    callback=iactions.delete_model,
                ),
            ],
        ),
    ]

    user_interface: ui.UserInterface = ui.UserInterface()

    ui.prompt(
        user_interface,
        DIOLAGUE_TREE,
        header=f" ------- Image Enhance Menu ------- (Session: '{user_interface.session.get_name_formatted()}')\n",
    )


if __name__ == "__main__":
    main()
