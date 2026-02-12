import click
import yaml

from waveorder.api._settings import MyBaseModel


def echo_settings(settings: MyBaseModel):
    click.echo(
        yaml.dump(
            settings.model_dump(), default_flow_style=False, sort_keys=False
        )
    )


def echo_headline(headline):
    click.echo(click.style(headline, fg="green"))
