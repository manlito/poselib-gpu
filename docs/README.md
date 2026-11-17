# Documentation site

Sphinx with MyST Markdown, built from this folder, with the site's own theme in `_theme/jpo` (just-matching's
theme, after just-ba's, with this project's accent and icon). After the pages, `conf.py` puts the live demo and
the benchmark next to them: `web/demo.html` and `web/benchmark.html` with the site's top bar as `demo/` and
`benchmark/`, the WebAssembly module in `lib/`, and the demo's images and matches in `samples/`. `_extra/` is
copied as it is: the pages' top-bar style and Cloudflare's `_headers`.

Rule for every page: what visitors read carries no references to the project's notes (PLAN.md, RESULTS.md,
milestones, commit hashes) and no links to the repository, which is private.

```bash
uv venv --python 3.12 docs/.venv && uv pip install --python docs/.venv/bin/python -r docs/requirements.txt
```

With a browser build in `build/wasm` (README.md, "Build"; `tools/fetch_webgpu.sh browser` first for WebGPU):

```bash
docs/.venv/bin/sphinx-build -W --keep-going -b html docs docs/_build/html
python3 -m http.server 8768 --bind 127.0.0.1 --directory docs/_build/html   # then open /demo/ in Chrome
```

`JPO_WASM_DIR` points at another build's `web/` folder. Without the module the build warns, so `-W` fails. The
site is about 1.7 MB.

The demo's data is in `web/samples`, made once by `tools/demo_data.py` with just-keypoints' and just-matching's
command-line tools; the site build does not need them.

## Building and deploying

The `site` job of the `docs` workflow (`.github/workflows/docs.yml`) is started by hand:

```bash
gh workflow run docs                    # build the module and the site
gh workflow run docs -f deploy=true     # the same, then publish to Cloudflare Pages
```

The site is uploaded as the run's `docs-html` artifact either way. Deploying needs the repository secrets
`CLOUDFLARE_API_TOKEN` (permission "Cloudflare Pages: Edit") and `CLOUDFLARE_ACCOUNT_ID`, and a Pages project
made for direct upload, named by the repository variable `CLOUDFLARE_PAGES_PROJECT` (default `poselib-gpu`).
From a machine where wrangler is logged in, the same upload is:

```bash
npx --yes wrangler@4.162.0 pages deploy docs/_build/html --project-name poselib-gpu --branch main
```
