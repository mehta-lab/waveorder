import click
import yaml


def echo_settings(settings):
    click.echo(yaml.dump(settings.dict(), default_flow_style=False, sort_keys=False))


def echo_headline(headline):
    click.echo(click.style(headline, fg="green"))
