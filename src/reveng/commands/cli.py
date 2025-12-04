import tyro
from reveng.commands.get_trajectory import get_trajectory

def main():
    tyro.extras.subcommand_cli_from_dict(
        {
            "get_trajectory": get_trajectory
        }
    )

if __name__ == "__main__":
    main()
