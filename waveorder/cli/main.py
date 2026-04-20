import importlib
import logging
import os
import warnings

# Suppress noisy CUDA and dependency warnings.
# PYTHONWARNINGS env var catches warnings from torch's C++ queued callbacks
# that fire after import but before Python filterwarnings can intercept them.
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::DeprecationWarning")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("iohub").setLevel(logging.ERROR)

import click

CONTEXT = {"help_option_names": ["-h", "--help"]}

# Registry of commands: name → "module.path:attribute"
# Modules are only imported when the command is actually invoked.
_LAZY_COMMANDS = {
    "simulate": "waveorder.cli.simulate:_simulate_cli",
    "reconstruct": "waveorder.cli.reconstruct:_reconstruct_cli",
    "compute-transfer-function": "waveorder.cli.compute_transfer_function:_compute_transfer_function_cli",
    "apply-inverse-transfer-function": "waveorder.cli.apply_inverse_transfer_function:_apply_inverse_transfer_function_cli",
    "view": "waveorder.cli.view:_view_cli",
}

# Optional commands that may not be installed
_OPTIONAL_LAZY_COMMANDS = {
    "interactive": "waveorder.cli.gui_widget:gui",
    "benchmark": "waveorder.cli.bench:benchmark",
}

_ALIASES = {
    "rec": "reconstruct",
    "sim": "simulate",
    "v": "view",
    "compute-tf": "compute-transfer-function",
    "apply-inv-tf": "apply-inverse-transfer-function",
    "gui": "interactive",
    "bm": "benchmark",
}

_DISPLAY_ORDER = [
    ("reconstruct", 0),
    ("compute-transfer-function", 2),
    ("apply-inverse-transfer-function", 2),
    ("simulate", 0),
    ("view", 0),
    ("interactive", 0),
    ("benchmark", 0),
]


def _import_command(import_path: str):
    """Import a click command from a 'module.path:attribute' string."""
    module_path, attr = import_path.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


class LazyAliasGroup(click.Group):
    """Click group that defers subcommand imports until invocation.

    Commands registered in ``_LAZY_COMMANDS`` are only imported when
    actually invoked, avoiding heavy imports (torch, numpy, etc.) for
    ``wo --help`` or unrelated subcommands.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases: dict[str, str] = {}
        self._reverse: dict[str, str] = {}
        self._display_order: list[tuple[str, int]] = []
        self._lazy: dict[str, str] = {}
        self._optional_lazy: dict[str, str] = {}

    def add_lazy(self, name: str, import_path: str):
        self._lazy[name] = import_path

    def add_optional_lazy(self, name: str, import_path: str):
        self._optional_lazy[name] = import_path

    def add_alias(self, alias: str, command_name: str):
        self._aliases[alias] = command_name
        self._reverse[command_name] = alias

    def get_command(self, ctx, cmd_name):
        # Resolve alias
        cmd_name = self._aliases.get(cmd_name, cmd_name)

        # Check if already loaded
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd

        # Lazy load
        import_path = self._lazy.get(cmd_name) or self._optional_lazy.get(cmd_name)
        if import_path is not None:
            try:
                cmd = _import_command(import_path)
                self.add_command(cmd, cmd_name)
                return cmd
            except Exception:
                return None

        return None

    def list_commands(self, ctx):
        # Registered + lazy (required) + optional (assume available for listing)
        names = set(self.commands.keys()) | set(self._lazy.keys()) | set(self._optional_lazy.keys())
        # Return in display order, then any extras
        ordered = [name for name, _ in self._display_order if name in names]
        for name in sorted(names - set(ordered)):
            ordered.append(name)
        return ordered

    def format_usage(self, ctx, formatter):
        formatter.write_usage("wo", "[OPTIONS] COMMAND [ARGS]...")

    def format_help(self, ctx, formatter):
        self.format_usage(ctx, formatter)
        formatter.write("\n")
        formatter.write("  waveorder [\033[92mwo\033[0m]: Wave-optical simulation and reconstruction\n")
        formatter.write("\n")

        formatter.write("Options:\n")
        formatter.write("  -h, --help  Show this message and exit.\n")
        formatter.write("\n")

        formatter.write("Commands:\n")
        col = 40
        available = set(self.list_commands(ctx))
        for name, indent in self._display_order:
            if name not in available:
                continue
            alias = self._reverse.get(name)
            pad = " " * indent
            label = f"  {pad}{name}"
            if alias:
                gap = " " * max(1, col + indent - len(label))
                label += f"{gap}[\033[92m{alias}\033[0m]"
            formatter.write(label + "\n")

        formatter.write("\n")


@click.group(context_settings=CONTEXT, cls=LazyAliasGroup)
def cli():
    pass


# Register lazy commands
for name, path in _LAZY_COMMANDS.items():
    cli.add_lazy(name, path)
for name, path in _OPTIONAL_LAZY_COMMANDS.items():
    cli.add_optional_lazy(name, path)

# Display order and aliases
cli._display_order = _DISPLAY_ORDER
for alias, target in _ALIASES.items():
    cli.add_alias(alias, target)

if __name__ == "__main__":
    cli()
