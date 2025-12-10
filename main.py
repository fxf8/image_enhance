import image_enhance.interface as ui
import image_enhance.actions as iactions


def main():
    DIOLAGUE_TREE: list[ui.MenuOption] = [
        ui.MenuOption(
            name="Manage Session",
            suboptions=[
                ui.MenuOption(
                    name="Save Session",
                    callback=iactions.save_session,
                ),
            ],
        ),
        ui.MenuOption(
            name="Manage Databases",
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
