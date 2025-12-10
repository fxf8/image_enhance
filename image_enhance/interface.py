from collections.abc import Callable
from dataclasses import dataclass

import image_enhance.session as isession


class UserInterface:
    session: isession.Session | None

    def __init__(self, session: isession.Session | None = None):
        self.location = []
        self.session = session


@dataclass
class MenuOption:
    name: str
    description: str | None = None
    header: str | None = None
    callback: Callable[[UserInterface], None] | None = None
    suboptions: list["MenuOption"] | None = None


def get_validated_input(
    prompt: str,
    validator: Callable[[str], bool],
    error_message: str | Callable[[str], str] | None = None,
    exit_string: str | None = "e",
) -> str | None:
    while True:
        user_input: str = input(prompt)

        if exit_string is not None and user_input == exit_string:
            return None

        if validator(user_input):
            return user_input

        if error_message is not None:
            if callable(error_message):
                print(error_message(user_input))

            else:
                print(error_message)


def prompt(
    ui: UserInterface, diolague_tree: list[MenuOption], header: str | None = None
):
    while True:
        if header is not None:
            print(header)

        for index, menu_option in enumerate(diolague_tree):
            if menu_option.description is not None:
                print(f"{index + 1}. {menu_option.name} - {menu_option.description}")

            else:
                print(f"{index + 1}. {menu_option.name}")

        print()

        selection: str | None = get_validated_input(
            "Select an option (or press e to exit): ",
            lambda x: x.isdigit() and 1 <= int(x) <= len(diolague_tree),
            error_message=f"\nInvalid selection. Please enter a number between 1 and {len(diolague_tree)} (inclusive)\n",
            exit_string="e",
        )

        print()

        if selection is None:
            return

        choice_number = int(selection) - 1

        if (callback := diolague_tree[choice_number].callback) is not None:
            callback(ui)

        if (suboptions := diolague_tree[choice_number].suboptions) is not None:
            prompt(ui, suboptions, header=diolague_tree[choice_number].header)
