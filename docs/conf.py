# Sphinx configuration of the documentation site. Build: see docs/README.md.
#
# Next to the pages, after the build, this file puts the live demo, the benchmark and what they load:
#   lib/        jpo.js and its WebAssembly module, from a browser build (JPO_WASM_DIR, default build/wasm/web)
#   demo/       web/demo.html with the site's top bar, and web/demo.js
#   benchmark/  web/benchmark.html with the site's top bar
#   samples/    the demo's images and matches (web/samples, made by tools/demo_data.py)
# A missing module is a warning, so `sphinx-build -W` fails without one.
import os
import shutil

from sphinx.util import logging

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

project = "poselib-gpu"
author = "the poselib-gpu authors"
copyright = "the poselib-gpu authors"
release = ""
version = ""

extensions = ["myst_parser"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", ".venv", "README.md", "_theme", "_extra"]

html_theme = "jpo"
html_theme_path = ["_theme"]
html_title = project
html_extra_path = ["_extra"]
html_copy_source = False
html_show_sourcelink = False

MODULE_FILES = ["jpo.js", "jpo_wasm.js", "jpo_wasm.wasm"]

ICON = ('<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linejoin="round"><path d="M5 19L2 13h6z"></path><path d="M17 19l-3-6h6z"></path>'
        '<path d="M5 13L11 4.5 17 13" stroke-dasharray="2 2"></path>'
        '<circle cx="11" cy="3.6" r="1.8" fill="currentColor" stroke="none"></circle></svg>')


def site_header(current):
    link = lambda href, text, page: (f'<a class="jk-toplink" href="{href}"' + (' aria-current="page"><strong>'
                                     f'{text}</strong></a>' if page == current else f'>{text}</a>'))
    return f"""<a class="jk-skip" href="#content">Skip to content</a>
<header class="jk-top">
  <div class="jk-top-in">
    <a class="jk-brand" href="../index.html">{ICON}<span>poselib-gpu</span></a>
    <span class="jk-grow"></span>
    {link("../index.html", "Documentation", "docs")}
    {link("../demo/index.html", "Live demo", "demo")}
    {link("../benchmark/index.html", "Benchmark", "benchmark")}
    <a class="jk-toplink jk-gh" href="https://github.com/manlito/poselib-gpu" title="poselib-gpu on GitHub" aria-label="poselib-gpu on GitHub"><svg width="20" height="20" viewBox="0 0 16 16" aria-hidden="true" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg></a>
  </div>
</header>"""


SITE_HEAD = """<link rel="stylesheet" href="../demo/site-bar.css">
<script type="application/json" id="demo-config">
{"lib": "../lib/", "samples": "../samples/"}
</script>"""


def site_page(page, name, title):
    """A page of web/ with the site's top bar and the places of what it loads."""
    for marker in ("<!-- site:head -->", "<!-- site:header -->"):
        if marker not in page:
            raise ValueError(f"web/{name}.html has no {marker}")
    page = page.replace("<!-- site:head -->", SITE_HEAD).replace("<!-- site:header -->", site_header(name))
    page = page.replace("<!-- site:refs -->", '<a href="../references.html">References and licences</a>.')
    start, end = page.index("<title>"), page.index("</title>")
    return page[:start] + f"<title>{title} · poselib-gpu" + page[end:]


def copy_web(app, exception):
    if exception or app.builder.format != "html":
        return
    log = logging.getLogger(__name__)
    out = app.outdir
    for d in ("lib", "demo", "benchmark", "samples"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    wasm = os.environ.get("JPO_WASM_DIR", os.path.join(ROOT, "build", "wasm", "web"))
    missing = [f for f in MODULE_FILES if not os.path.isfile(os.path.join(wasm, f))]
    if missing:
        log.warning("the demo needs a browser build of the library; not found in %s: %s (set JPO_WASM_DIR)",
                    wasm, ", ".join(missing))
    else:
        for f in MODULE_FILES:
            shutil.copyfile(os.path.join(wasm, f), os.path.join(out, "lib", f))
        log.info("module copied from %s", wasm)

    web = os.path.join(ROOT, "web")
    for name, title in (("demo", "Live demo"), ("benchmark", "Benchmark")):
        with open(os.path.join(web, f"{name}.html"), encoding="utf-8") as f:
            page = site_page(f.read(), name, title)
        with open(os.path.join(out, name, "index.html"), "w", encoding="utf-8") as f:
            f.write(page)
    shutil.copyfile(os.path.join(web, "demo.js"), os.path.join(out, "demo", "demo.js"))
    samples = os.path.join(web, "samples")
    for f in sorted(os.listdir(samples)):
        if f.endswith((".jpg", ".json")):
            shutil.copyfile(os.path.join(samples, f), os.path.join(out, "samples", f))


def setup(app):
    app.connect("build-finished", copy_web)
