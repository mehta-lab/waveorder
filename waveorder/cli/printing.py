import typer
import yaml

from waveorder.api._settings import MyBaseModel


def echo_settings(settings: MyBaseModel):
    typer.echo(yaml.dump(settings.model_dump(), default_flow_style=False, sort_keys=False))


def echo_headline(headline):
    typer.echo(typer.style(headline, fg=typer.colors.GREEN))
