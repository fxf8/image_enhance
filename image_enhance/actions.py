import pathlib

import image_enhance.interface as ui


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
