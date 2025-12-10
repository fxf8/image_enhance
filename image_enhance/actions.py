import fnmatch
import pathlib

import image_enhance.interface as ui
import image_enhance.session as isession


def save_session(user_interface: ui.UserInterface):
    def validator(session_path_string: str) -> bool:
        if session_path_string == "":
            return False

        session_path: pathlib.Path = pathlib.Path(session_path_string)

        return not session_path.is_dir()

    session_path_string: str | None = ui.get_validated_input(
        "Please provide a path which to save the session (or press e to exit): ",
        validator,
        exit_string="e",
        error_message="\nError, save path must be non-empty and not a directory\n",
    )

    if session_path_string is None:
        return

    session_path: pathlib.Path = pathlib.Path(session_path_string)

    if session_path.exists():
        overwrite: str | None = ui.get_validated_input(
            f"File already exists at {session_path}. Overwrite? (y/n)",
        )

        if overwrite != "y":
            print(f"Aborted save. Session NOT saved to path '{session_path}'")

            return

    user_interface.session.save_session(session_path)

    print(f"Session saved successfully to path '{session_path}'")


def load_session(user_interface: ui.UserInterface):
    def validator(session_path_string: str) -> bool:
        if session_path_string == "":
            return False

        session_path: pathlib.Path = pathlib.Path(session_path_string)

        return session_path.is_file()

    session_path_string: str | None = ui.get_validated_input(
        "Please provide a path from which to load the session (or press e to exit): ",
        validator,
        exit_string="e",
        error_message="\nError, load path must be a valid file\n",
    )

    if session_path_string is None:
        return

    session_path: pathlib.Path = pathlib.Path(session_path_string)

    user_interface.session = isession.Session.load_session(session_path)

    print(f"Session loaded successfully from path '{session_path}'")


def _select_database(
    user_interface: ui.UserInterface, prompt_message: str
) -> str | None:
    db_names = list(user_interface.session.databases.keys())
    if not db_names:
        print("No databases available.")
        return None

    print("\nAvailable databases:")
    for i, name in enumerate(db_names):
        print(f"{i + 1}. {name}")
    print()

    selection = ui.get_validated_input(
        prompt_message,
        lambda x: x.isdigit() and 1 <= int(x) <= len(db_names),
        error_message=f"\nInvalid selection. Please enter a number between 1 and {len(db_names)}.\n",
        exit_string="e",
    )

    if selection is None:
        return None

    return db_names[int(selection) - 1]


def list_databases(user_interface: ui.UserInterface):
    databases = user_interface.session.databases
    if not databases:
        print("No databases found.")
        return

    print("\nAvailable databases:")
    for name, db in databases.items():
        print(f"- {name}: {len(db)} images")
    print()


def create_dataset(user_interface: ui.UserInterface):
    dataset_name = ui.get_validated_input(
        "Enter a name for the new, empty dataset: ",
        lambda x: x != "" and x not in user_interface.session.databases,
        error_message="\nDataset name cannot be empty or already exist.\n",
        exit_string="e",
    )
    if dataset_name is None:
        return

    user_interface.session.databases[dataset_name] = []
    print(f"\nCreated empty dataset '{dataset_name}'.\n")


def import_from_glob(user_interface: ui.UserInterface):
    db_name = ui.get_validated_input(
        "Enter a name for the new or existing database: ",
        lambda x: x != "",
        error_message="\nDatabase name cannot be empty.\n",
        exit_string="e",
    )
    if db_name is None:
        return

    glob_pattern = ui.get_validated_input(
        "Enter the glob pattern for images to import (e.g., 'images/*.jpg'): ",
        lambda x: x != "",
        error_message="\nGlob pattern cannot be empty.\n",
        exit_string="e",
    )
    if glob_pattern is None:
        return

    user_interface.session.import_glob(db_name, glob_pattern)
    count = len(user_interface.session.databases.get(db_name, []))
    print(f"\nImported images into database '{db_name}'. It now has {count} images.\n")


def delete_database(user_interface: ui.UserInterface):
    db_name_to_delete = _select_database(
        user_interface, "Select a database to delete (or press e to exit): "
    )
    if db_name_to_delete is None:
        return

    confirm = ui.get_validated_input(
        f"Are you sure you want to delete database '{db_name_to_delete}'? This cannot be undone. (y/n): ",
        lambda x: x.lower() in ["y", "n"],
        exit_string="e",
    )

    if confirm == "y":
        user_interface.session.delete_database(db_name_to_delete)
        print(f"Database '{db_name_to_delete}' deleted.")
    else:
        print("Deletion aborted.")


def copy_database(user_interface: ui.UserInterface):
    source_db = _select_database(user_interface, "Select the source database to copy: ")
    if source_db is None:
        return

    target_db_name = ui.get_validated_input(
        "Enter the name for the new copied database: ",
        lambda x: x != "" and x not in user_interface.session.databases,
        error_message="\nDatabase name cannot be empty or already exist.\n",
        exit_string="e",
    )
    if target_db_name is None:
        return

    user_interface.session.copy_database(source_db, target_db_name)
    print(f"Database '{source_db}' copied to '{target_db_name}'.")


def split_database(user_interface: ui.UserInterface):
    source_db = _select_database(user_interface, "Select the database to split: ")
    if source_db is None:
        return

    try:
        ratio_str = ui.get_validated_input(
            "Enter the split ratio for the first database (0.0 to 1.0): ",
            lambda x: 0.0 < float(x) < 1.0,
            error_message="\nRatio must be a number between 0 and 1 (exclusive).\n",
            exit_string="e",
        )
        if ratio_str is None:
            return
        split_ratio = float(ratio_str)
    except ValueError:
        print("\nInvalid input. Please enter a number.\n")
        return

    first_db_name = ui.get_validated_input(
        "Enter name for the first new database: ", lambda x: x != "", exit_string="e"
    )
    if first_db_name is None:
        return

    second_db_name = ui.get_validated_input(
        "Enter name for the second new database: ", lambda x: x != "", exit_string="e"
    )
    if second_db_name is None:
        return

    if (
        first_db_name in user_interface.session.databases
        or second_db_name in user_interface.session.databases
    ):
        print("\nOne of the specified database names already exists. Aborting.\n")
        return

    user_interface.session.split_database(
        source_db, split_ratio, first_db_name, second_db_name
    )
    print(
        f"Database '{source_db}' split into '{first_db_name}' and '{second_db_name}'."
    )
    del user_interface.session.databases[source_db]
    print(f"Original database '{source_db}' was removed.")


def merge_databases(user_interface: ui.UserInterface):
    db_names = list(user_interface.session.databases.keys())
    if len(db_names) < 2:
        print("Need at least two databases to merge.")
        return

    print("\nAvailable databases:")
    for i, name in enumerate(db_names):
        print(f"{i + 1}. {name}")
    print()

    selections_str = ui.get_validated_input(
        "Enter the numbers of databases to merge, separated by commas (e.g., 1,3): ",
        lambda s: all(
            x.strip().isdigit() and 1 <= int(x.strip()) <= len(db_names)
            for x in s.split(",")
        )
        and len(set(s.split(","))) > 1,
        error_message="\nPlease enter at least two unique, valid numbers separated by commas.\n",
        exit_string="e",
    )
    if selections_str is None:
        return

    selected_indices = [int(x.strip()) - 1 for x in selections_str.split(",")]
    dbs_to_merge = [db_names[i] for i in selected_indices]

    merged_db_name = ui.get_validated_input(
        "Enter the name for the new merged database: ",
        lambda x: x != "",
        exit_string="e",
    )
    if merged_db_name is None:
        return

    if merged_db_name in user_interface.session.databases:
        print(f"\nDatabase '{merged_db_name}' already exists. Aborting.\n")
        return

    user_interface.session.merge_databases(merged_db_name, dbs_to_merge)
    print(f"Databases {dbs_to_merge} merged into '{merged_db_name}'.")


def corrupt_database(user_interface: ui.UserInterface):
    db_name = _select_database(user_interface, "Select a database to corrupt: ")
    if db_name is None:
        return

    image_glob = ui.get_validated_input(
        "Enter image name glob pattern to corrupt (or '*' for all): ",
        lambda x: x != "",
        exit_string="e",
    )
    if image_glob is None:
        return

    try:
        iterations_str = ui.get_validated_input(
            "Enter number of corruption iterations (e.g., 1-8): ",
            lambda x: x.isdigit() and int(x) > 0,
            error_message="\nPlease enter a positive number.\n",
            exit_string="e",
        )
        if iterations_str is None:
            return
        iterations = int(iterations_str)
    except ValueError:
        print("\nInvalid input. Please enter a number.\n")
        return

    user_interface.session.corrupt_images(db_name, image_glob, iterations)
    print(f"Corrupted images matching '{image_glob}' in database '{db_name}'.")


def show_images_in_database(user_interface: ui.UserInterface):
    db_name = _select_database(
        user_interface, "Select a database to show images from: "
    )
    if db_name is None:
        return

    image_glob = ui.get_validated_input(
        "Enter image name glob pattern to show (or '*' for all): ",
        lambda x: x != "",
        exit_string="e",
    )
    if image_glob is None:
        return

    print("\nDisplaying images... Close the plot window to continue.")
    user_interface.session.show_images(db_name, image_glob)


def export_images(user_interface: ui.UserInterface):
    db_name = _select_database(
        user_interface, "Select a database to export images from: "
    )
    if db_name is None:
        return

    image_glob_input = ui.get_validated_input(
        "Enter image name glob pattern to export (or '*' for all, default: '*'): ",
        lambda x: True,  # any string is fine
        exit_string="e",
    )
    if image_glob_input is None:
        return
    image_glob = image_glob_input if image_glob_input != "" else "*"

    export_path_str = ui.get_validated_input(
        "Enter the directory path to export to: ",
        lambda x: pathlib.Path(x).is_dir(),
        error_message="\nPath must be an existing directory.\n",
        exit_string="e",
    )
    if export_path_str is None:
        return
    export_path = pathlib.Path(export_path_str)

    export_choice = ui.get_validated_input(
        "Export original, fixed, or both? (o/f/b, default: b): ",
        lambda x: x.lower() in ["o", "f", "b", ""],
        exit_string="e",
    )
    if export_choice is None:
        return
    export_choice = export_choice.lower() if export_choice != "" else "b"

    images_to_export = [
        img
        for img in user_interface.session.databases[db_name]
        if fnmatch.fnmatch(img.image_path.name, image_glob)
    ]

    if not images_to_export:
        print("No images found matching the pattern.")
        return

    exported_count = 0
    for i, image_sample in enumerate(images_to_export):
        original_stem = image_sample.image_path.stem
        original_suffix = image_sample.image_path.suffix

        if export_choice in ["o", "b"]:
            export_filename = f"{original_stem}_original_{i}{original_suffix}"
            image_sample.export_original_image(export_path / export_filename)
            exported_count += 1

        if export_choice in ["f", "b"]:
            if image_sample.fixed_image_tensor is not None:
                export_filename = f"{original_stem}_fixed_{i}{original_suffix}"
                image_sample.export_fixed_image(export_path / export_filename)
                exported_count += 1
            elif export_choice == "f":
                print(
                    f"Skipping '{image_sample.image_path.name}' - no fixed version available."
                )

    print(f"\nExported {exported_count} file(s) to '{export_path}'.\n")


def create_model(user_interface: ui.UserInterface):
    model_name = ui.get_validated_input(
        "Enter a name for the new model: ",
        lambda x: x != "" and x not in user_interface.session.models,
        error_message="\nModel name cannot be empty or already exist.\n",
        exit_string="e",
    )
    if model_name is None:
        return

    user_interface.session.create_model(model_name)
    print(f"\nModel '{model_name}' created successfully.\n")


def _select_model(user_interface: ui.UserInterface, prompt_message: str) -> str | None:
    model_names = list(user_interface.session.models.keys())
    if not model_names:
        print("No models available.")
        return None

    print("\nAvailable models:")
    for i, name in enumerate(model_names):
        print(f"{i + 1}. {name}")
    print()

    selection = ui.get_validated_input(
        prompt_message,
        lambda x: x.isdigit() and 1 <= int(x) <= len(model_names),
        error_message=f"\nInvalid selection. Please enter a number between 1 and {len(model_names)}.\n",
        exit_string="e",
    )

    if selection is None:
        return None

    return model_names[int(selection) - 1]


def list_models(user_interface: ui.UserInterface):
    models = user_interface.session.models
    if not models:
        print("No models found.")
        return

    print("\nAvailable models:")
    for name in models.keys():
        history = user_interface.session.model_training_histories.get(name)
        trained_batches = len(history.losses) if history else 0
        print(f"- {name} (Trained for {trained_batches} batches)")
    print()


def delete_model(user_interface: ui.UserInterface):
    model_name = _select_model(user_interface, "Select a model to delete: ")
    if model_name is None:
        return

    confirm = ui.get_validated_input(
        f"Are you sure you want to delete model '{model_name}'? (y/n): ",
        lambda x: x.lower() in ["y", "n"],
    )

    if confirm == "y":
        user_interface.session.delete_model(model_name)
        print(f"Model '{model_name}' deleted.")
    else:
        print("Deletion aborted.")


def train_model(user_interface: ui.UserInterface):
    model_name = _select_model(user_interface, "Select a model to train: ")
    if model_name is None:
        return

    db_name = _select_database(user_interface, "Select a database for training: ")
    if db_name is None:
        return

    # Helper for getting numeric input
    def get_numeric_input(prompt_text, validation, default_val):
        val_str = ui.get_validated_input(
            f"{prompt_text} (default: {default_val}): ",
            lambda x: x == "" or validation(x),
            exit_string="e",
        )
        if val_str is None:
            return None  # User exited
        return default_val if val_str == "" else type(default_val)(val_str)

    maximum_training_samples = get_numeric_input(
        "Maximum Training Samples", lambda x: x.isdigit() and int(x) > 0, 30_000
    )
    if maximum_training_samples is None:
        return

    epochs = get_numeric_input("Epochs", lambda x: x.isdigit() and int(x) > 0, 1)
    if epochs is None:
        return

    batch_size = get_numeric_input(
        "Batch Size", lambda x: x.isdigit() and int(x) > 0, 32
    )
    if batch_size is None:
        return

    learning_rate = get_numeric_input("Learning Rate", lambda x: float(x) > 0, 1e-3)
    if learning_rate is None:
        return

    sample_expansion = get_numeric_input(
        "Sample Expansion", lambda x: x.isdigit() and int(x) > 0, 10
    )
    if sample_expansion is None:
        return

    maximum_dimension = get_numeric_input(
        "Max Dimension", lambda x: x.isdigit() and int(x) > 0, 256
    )
    if maximum_dimension is None:
        return

    split_resize_ratio = get_numeric_input(
        "Split/Resize Ratio", lambda x: 0.0 <= float(x) <= 1.0, 0.5
    )
    if split_resize_ratio is None:
        return

    display_progress_str = ui.get_validated_input(
        "Display progress? (y/n, default: y): ", lambda x: x.lower() in ["y", "n", ""]
    )
    if display_progress_str is None:
        return
    display_progress = display_progress_str.lower() != "n"

    print("\nStarting model training... This may take a while.")
    user_interface.session.train_model(
        model_name=model_name,
        database_name=db_name,
        maximum_training_samples=maximum_training_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sample_expansion=sample_expansion,
        display_progress=display_progress,
        maximum_dimension=maximum_dimension,
        split_resize_ratio=split_resize_ratio,
    )
    print(f"\nFinished training model '{model_name}'.")


def use_model(user_interface: ui.UserInterface):
    model_name = _select_model(user_interface, "Select a model to use: ")
    if model_name is None:
        return

    db_name = _select_database(
        user_interface, "Select a database to apply the model to: "
    )
    if db_name is None:
        return

    image_glob = ui.get_validated_input(
        "Enter image name glob pattern to apply model to (or '*' for all, default: '*'): ",
        lambda x: x != "",
    )
    if image_glob is None:
        return
    if image_glob == "":
        image_glob = "*"

    # Optional resize
    new_size = None
    if (
        ui.get_validated_input(
            "Resize output? (y/n, default: n): ", lambda x: x.lower() in ["y", "n", ""]
        )
        == "y"
    ):
        width_str = ui.get_validated_input(
            "New width: ", lambda x: x.isdigit() and int(x) > 0
        )
        if width_str is None:
            return
        height_str = ui.get_validated_input(
            "New height: ", lambda x: x.isdigit() and int(x) > 0
        )
        if height_str is None:
            return
        new_size = (int(height_str), int(width_str))  # H, W

    print("\nApplying model to images...")
    success = user_interface.session.use_model(
        model_name=model_name,
        database_name=db_name,
        image_glob=image_glob,
        new_size=new_size,
    )
    if success:
        print(
            "Model applied successfully. You can now view the results with 'Show Images'."
        )
    else:
        print("Failed to apply model. Check if model and database exist.")


def plot_model_training_history(user_interface: ui.UserInterface):
    model_name = _select_model(user_interface, "Select a model to plot history for: ")
    if model_name is None:
        return

    print("\nDisplaying plot... Close the plot window to continue.")
    success = user_interface.session.plot_model_training_history(model_name)
    if not success:
        print("Could not plot history. Does the model have any training history?")
