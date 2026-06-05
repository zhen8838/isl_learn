from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def render_template(template_name: str, model: object) -> str:
    env = Environment(
        loader=FileSystemLoader(Path("./etensor/templates")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name).render(**model.__dict__)


def emit_source(source: str, prefix: Path) -> Path:
    output = prefix.with_suffix(".cu")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    return output


def build_shared_library(prefix: Path, *, python: str = sys.executable) -> Path:
    os.environ.setdefault("TMPDIR", "./tmp/etensor_tutorial/tilelang_tmp")
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "make",
            "-C",
            "etensor",
            f"PYTHON={python}",
            f"PREFIX=../{prefix}",
        ],
        check=True,
    )
    return prefix.with_suffix(".so")
