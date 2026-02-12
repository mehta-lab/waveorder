# Contributing guide

Thanks for your interest in contributing to `waveorder`!

Please see the following steps for our workflow.

## Getting started

Please read the [README](./README.md) for an overview of the project,
and how you can install and use the package.

## Issues

We use [issues](https://github.com/mehta-lab/waveorder/issues) to track
bug reports, feature requests, and provide user support.

Before opening a new issue, please first search existing issues (including closed ones),
to see if there is an existing discussion about it.

### Setting up development environment

Install [Git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/getting-started/installation/).

If you have push permission to the repository,
clone the repository (the code blocks below are shell commands):

```sh
cd # to the directory you want to work in
git clone https://github.com/mehta-lab/waveorder.git
```

Otherwise, you can follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/mehta-lab/waveorder/fork) the repository.

Next, create a virtual environment and install the package with all dev dependencies:

```sh
cd waveorder/
uv sync --group dev --extra visual
```

Check that the tests pass:

```sh
uv run pytest
```

Finally, make the changes and [track them with Git](https://docs.github.com/en/get-started/using-git/about-git#example-contribute-to-an-existing-repository).

**Dependency groups:**

- `--group test` — pytest, hypothesis, napari test deps
- `--group docs` — sphinx and documentation deps
- `--group dev` — includes both `test` and `docs`
- `--extra visual` — napari, PyQt6, and visualization deps

### Code style

We use [pre-commit](https://pre-commit.com/) to lint with [ruff](https://docs.astral.sh/ruff/) automatically prior to each commit. Install pre-commit in your environment:

```sh
pre-commit install
```

To manually check formatting and linting:

```sh
uvx ruff check .
uvx ruff format --check .
```

### Developing documentation

Documentation infrastructure is built using [Markdown (.md)](https://www.sphinx-doc.org/en/master/usage/markdown.html) and [reStructuredText (.rst)](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) files. [Sphinx](https://www.sphinx-doc.org/en/master/index.html) utilizes these to build highly customization .html pages for User Guides, API References and Examples rendered directly from source code files.

#### Building the HTML version locally

```sh
cd waveorder/
uv sync --group docs
cd docs/
uv run sphinx-build -M html ./ ./build
```

Generated HTML documentation can be found in the ``build/html`` directory. Open ``build/html/index.html`` to view the home page for the documentation.

Automated building of `docs` is done via _read-the-docs_ which will automatically pull commits to the `main` branch and host them here: <https://waveorder.readthedocs.io>

#### Documentation change

If you find that any documentation in this project is incomplete, inaccurate, or ambiguous, please open an issue.
We welcome contributions to the documentation from users, particularly user guides that we can collaboratively edit.
