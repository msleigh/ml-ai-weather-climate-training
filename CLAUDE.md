# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a training course repository for "ML and AI for Weather, Climate and Environment" (ECMWF / University of Reading / DWD). The README.md is the primary course document — it contains extensive lecture notes, not just setup instructions. The actual course materials (slides, notebooks, code exercises) live in the `e-ai_ml2` git submodule.

## Setup

```bash
git submodule update --init --recursive   # fetch course materials
uv venv                                    # create venv (Python 3.11)
source .venv/bin/activate
uv sync                                    # install deps from pyproject.toml
```

Python version is pinned to 3.11 (`.python-version` and `pyproject.toml`).

## Key Dependencies

PyTorch, NumPy, Matplotlib, JupyterLab, eccodes (GRIB/BUFR data), requests, torchvision. All managed via `uv` and `pyproject.toml` (no `requirements.txt` at the top level).

## Repository Structure

- `README.md` — comprehensive lecture notes (1300+ lines), not just a project readme. This is the main authored content of this repo.
- `e-ai_ml2/` — git submodule pointing to `github.com/eumetnet-e-ai/e-ai_ml2`. Contains the upstream course code, notebooks, slides (LaTeX), and images. Do not modify files within this submodule.
- `notebooks/` — local scratch notebooks.
- `docker-test/` — Docker examples for lectures (Dockerfiles, test scripts, sample plots). The `03_dockerfile.txt` is built by CI.
- `assets/images/` — images referenced from README.md.

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- Trailing whitespace, end-of-file fixer, mixed line endings (enforces LF)
- Large file check, merge conflict check
- YAML and TOML validation
- **pymarkdown** linter on all Markdown files (MD022 and MD012 disabled; line length limit 192 chars; indent 4 spaces for MD007)
- **nbstripout** strips output from Jupyter notebooks before commit

## CI

Single GitHub Actions workflow (`.github/workflows/docker-03-plot.yml`): builds and pushes a Docker image from `docker-test/03_dockerfile.txt` to GHCR on pushes that touch `docker-test/` files.

## Editing Guidelines

- The README.md follows strict Markdown linting rules enforced by pymarkdown via pre-commit. Keep lines under 192 characters and use 4-space indentation for nested lists.
- GRIB data files (`*.grib2`, `*.grib2.bz2`) are gitignored.
- The `e-ai_ml2` submodule is upstream course material — changes belong in the upstream repo, not here.
