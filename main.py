import image_enhance.interface as ui


def main():
    DIOLAGUE_TREE: list[ui.MenuOption] = [
        ui.MenuOption(
            name="Merge Databases",
            description="Merge multiple databases into one",
            header="Merge Databases",
        ),
    ]

    ui.prompt(
        ui.UserInterface(),
        DIOLAGUE_TREE,
        header=" ------- Image Enhance Menu -------\n",
    )


if __name__ == "__main__":
    main()
