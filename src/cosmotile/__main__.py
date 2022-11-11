"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Cosmotile."""


if __name__ == "__main__":
    main(prog_name="cosmotile")  # pragma: no cover
