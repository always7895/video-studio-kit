# video-studio-kit

A config-driven kit for local video production on [ComfyUI](https://github.com/Comfy-Org/ComfyUI).
It turns pinned ComfyUI API graphs plus small YAML recipes into reproducible graphs, checks your
setup offline, and ships advisory QA helpers. It never approves anything: every output stays
**HOLD** until a person signs off.

## What it is

- `vsk.core`: safe YAML loader, SHA-256-pinned graph builder, offline doctor, loopback-only
  ComfyUI client (standard library only). Graphs are built, never submitted, by the kit.
- `config/recipes`: three example recipes (P-H3 still, MOSS still, Pika H3 Ref2VA video). Each
  binds named parameters to nodes of a pinned workflow JSON in `workflows/api`.
- `vsk.qa`: ArcFace identity screening, background optical flow, cut/freeze/jump checks, half-frame-rate detection.
- `vsk.post.master`: two-pass EBU R128 loudness mastering through ffmpeg.
- `setup/`: pinned ComfyUI and public-node installer, model manifest with sizes, SHA-256 and licences, downloader.

## Three project types

Examples live in `config/projects`; copy one and edit the steps.

| File | Type | Steps |
|---|---|---|
| `ad.yaml` | product ad | hero still, then image-to-video motion |
| `explainer.yaml` | economics explainer | abstract backdrop still |
| `documentary.yaml` | history documentary | establishing still, then motion |

The examples use placeholder paths and generic prompts only.

## Install from zero

```bash
git clone <this repository> video-studio-kit
cd video-studio-kit
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[qa]"          # core needs only PyYAML; [qa] adds numpy, opencv, insightface
pip install -e ".[test]"        # pytest
```

`insightface` may need a C++ toolchain on Windows. If its wheel fails, install the core
(`pip install -e .`) and add `numpy opencv-python` by hand; only `ArcFaceEngine` needs insightface.

Install ComfyUI and the public nodes at the pinned commits, then its own dependencies in your venv
(use the torch build that matches your GPU):

```bash
python setup/install_comfyui.py --dest /path/to/ComfyUI
pip install -r /path/to/ComfyUI/requirements.txt
```

Fetch model weights (large: check sizes first):

```bash
python setup/download_models.py --dry-run                      # HEAD check of every URL and size
python setup/download_models.py --models-dir /path/to/models   # download with SHA-256 verification
python setup/download_models.py --models-dir /path/to/models --verify
```

Gated or region-restricted items (FLUX.3 Klein, MiniMax H3 regional terms) are manual steps; see
`setup/models.yaml` and `NOTICE.md`.

Configure your machine. All paths come from `config/local.yaml` (gitignored) or environment
variables; nothing is hardcoded:

```bash
cp config/local.yaml.example config/local.yaml
# edit it, or set e.g. VSK_TOOLS_FFMPEG, VSK_PATHS_MODELS_ROOT, VSK_PATHS_INSIGHTFACE_ROOT
```

## Usage

```bash
python -m vsk doctor                # offline check of recipes, projects, registry, QA config
python -m vsk projects              # list project configs
python -m vsk build ad --out ./out  # build API graphs for every step (create-only files)
python -m pytest -q                 # CPU tests, no GPU, no ComfyUI, no home-directory access
```

Python API:

```python
from vsk.core import build_graph, build_project
graph = build_graph("ph3", {"prompt": "A glass jar on a stone table", "seed": 7})
graphs = build_project("documentary")   # {step_name: api_graph}
```

Submitting a graph to a running ComfyUI is left to you. Check `require_idle(client.queue_snapshot())`
first, hold your own GPU lease, and review the result.

To use your own workflow: export the API-format JSON into `workflows/api`, add a recipe YAML
that maps parameter names to `{node, input}` pairs, and pin the file's SHA-256. A modified graph with a
stale pin is refused.

## HOLD and human sign-off

Automated checks here are screening aids. Identity cosine, optical flow, cut and loudness numbers
flag items for review; they do not pass or fail a deliverable. Every result carries
`disposition: HOLD`, `config/qa.yaml` sets `automatic_approval: false`, and nothing in the kit
writes an approval. Publishing, rights, and compliance decisions belong to a human reviewer.

## Licence

MIT for this repository. Third-party software and model weights keep their own licences; see `NOTICE.md`.

## 繁體中文簡介

video-studio-kit 是以設定檔驅動的本機 ComfyUI 影片製作工具組：用 YAML 配方與固定 SHA-256 的 API 工作流產生可重現的 graph，離線檢查環境，並附 ArcFace 身份、背景光流、剪點與響度等輔助檢查。

- 安裝：建立 venv，`pip install -e ".[qa]"`，執行 `setup/install_comfyui.py` 安裝固定版本的 ComfyUI 與公開節點，`setup/download_models.py --dry-run` 先檢查模型網址與大小，再下載並驗證 SHA-256，最後複製 `config/local.yaml.example` 為 `config/local.yaml` 並填入本機路徑。
- 使用：`python -m vsk doctor` 離線檢查，`python -m vsk build ad` 建立工作流，`python -m pytest -q` 跑 CPU 測試。
- 原則：所有輸出預設為 HOLD。工具只做輔助篩檢，不會自動放行；放行、版權與法遵由人工簽核。
- 授權：本專案為 MIT；第三方軟體與模型各依其授權，請見 `NOTICE.md`。受限模型（FLUX.3 Klein、MiniMax H3 地區條款）需自行確認並手動取得。
