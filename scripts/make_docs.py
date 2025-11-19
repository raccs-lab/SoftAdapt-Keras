from pathlib import Path
import os

from pdoc import pdoc
from pdoc import render

out = Path("./docs")

if not out.exists():
    os.makedirs(out)

# Render parts of pdoc's documentation into docs/api...
render.configure(template_directory="./docs")
pdoc("softadapt", "!softadapt.__version__", output_directory=out)

# ...and rename the .html files to .md so that mkdocs picks them up!
for f in out.glob("**/*.html"):
    f.rename(f.with_suffix(".md"))

try:
    os.remove("./docs/index.md")
    os.remove("./docs/search.js")
except OSError:
    pass
