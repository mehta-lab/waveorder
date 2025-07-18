import click
import yaml
import threading
from waveorder.cli import jobs_mgmt

JM = jobs_mgmt.JobsManagement()


def echo_settings(settings, unique_id=""):
    if JM.do_print:
        click.echo(
            yaml.dump(
                settings.dict(), default_flow_style=False, sort_keys=False
            )
        )
    else:
        UpdateThread(
            uID=unique_id,
            msg=yaml.dump(
                settings.dict(), default_flow_style=False, sort_keys=False
            ),
        )


def echo_headline(headline, unique_id=""):
    if JM.do_print:
        click.echo(click.style(headline, fg="green"))
    else:
        UpdateThread(uID=unique_id, msg=headline)


def echo_text(text, unique_id=""):
    if JM.do_print:
        click.echo(text)
    else:
        UpdateThread(uID=unique_id, msg=text)


def UpdateThread(uID, msg):
    if uID == "":
        return
    msg = "Processing: " + msg.replace("\n", " ").replace("\\", "/")
    threading.Thread(
        target=JM.put_Job_in_list,
        args=(
            uID,
            msg,
        ),
    ).start()
