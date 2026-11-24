# Changelog

## 0.2.0

### Breaking
- No built-in face threshold. The hard-coded 0.65 is removed from code, `config/qa.yaml`
  and the example projects. `score_identity` and `score_frames` need an explicit threshold
  and raise `ThresholdNotConfigured` otherwise. Set `qa.face_cosine_min` (and optionally
  `qa.face_cosine_min_oblique`) in your project YAML; the templates ship with `null`.
- `requires-python` is now `>=3.12` (CI tests 3.12 and 3.13).

### Added
- Pluggable face backends: `insightface` (default) and `yunet_sface` (OpenCV Zoo, Apache-2.0 / MIT).
- `vsk qa calibrate-face`: recommends a threshold at FPR <= 1% from your own pairs, needs
  >= 50 positive and >= 300 negative pairs, reports Wilson 95% intervals, refuses otherwise.
  Advisory only; every result stays HOLD.
- Overrides outside the repository: `VSK_LOCAL_CONFIG`, `VSK_PROJECTS_DIR` (and
  `--local-config`, `--projects-dir`). Precedence: CLI > env > repo templates.
  `vsk doctor` shows which source is used. Do not paste its output into public issues.
- `vsk build --out` refuses a directory inside the repository.
- YuNet and SFace ONNX entries in `setup/models.yaml` (manual download, size and SHA-256 recorded).

### Changed
- ComfyUI-Manager is no longer a required custom node (optional, install by hand).
- CI: actions pinned by commit SHA (checkout v7.0.1, setup-python v7.1.0), Python 3.12 and 3.13 matrix,
  non-blocking job for newest numpy/opencv/insightface, comfyui-smoke stays on 3.12 (installs only `.[test]`).
- QA extra uses `opencv-python-headless`.

### Notes
- FLUX.3 Klein remains a candidate, not a recipe: it must first pass on an unpatched pinned ComfyUI.
  Licence text in `NOTICE.md` is the FLUX Kommunity License.
- Tests cover interface and threshold logic only. Nothing here approves a deliverable.
