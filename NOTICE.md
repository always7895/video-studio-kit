# Third-party notices

video-studio-kit is MIT licensed (see LICENSE). It bundles no model weights, no
ComfyUI source and no custom-node source. The scripts under `setup/` fetch third-party
software and weights from their upstream owners; each remains under its own licence.
You are responsible for reading and accepting the terms before you download or use them.

## Software installed by `setup/install_comfyui.py`

| Component | Licence | Source |
|---|---|---|
| ComfyUI v0.46.0 | GPL-3.0 | https://github.com/Comfy-Org/ComfyUI |
| ComfyUI-Inpaint-CropAndStitch | GPL-3.0 | https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch |
| ComfyUI Nvidia VFX nodes (Nvidia_RTX_Nodes_ComfyUI) | Apache-2.0 | https://github.com/Comfy-Org/ComfyUI_Nvidia_VFX_Nodes |

These are cloned, not redistributed. Verify each licence in the cloned tree. ComfyUI-Manager is deliberately not installed (it self-updates at run time, which conflicts with pinning); install it by hand if you want it.

## Python dependencies

| Package | Licence |
|---|---|
| PyYAML | MIT |
| numpy (qa extra) | BSD-3-Clause |
| opencv-python-headless (qa extra) | Apache-2.0 (OpenCV), MIT (wrapper) |
| insightface (qa extra) | MIT for the code; the pretrained `buffalo_l` / ArcFace model packs are for non-commercial research use only, per the InsightFace project |
| pytest (test extra) | MIT |

## Model weights listed in `setup/models.yaml`

| Model files | Licence | Notes |
|---|---|---|
| `minimax_h3_*`, `pika_h3_flex_*`, `qwen3vl_32b_minimax_h3_*` (Comfy-Org/MiniMax-H3) | MiniMax H3 Community License Agreement | Regional terms: the grant covers the "Applicable Territory", defined as worldwide excluding the European Union, the United Kingdom, the Republic of Korea and the United States of America. Hosted-service and attribution clauses apply. Read the full text at https://huggingface.co/MiniMaxAI/MiniMax-H3 before use. |
| `p-h3-image_*` (PrunaAI/P-Image-OSS-Comfy) | MiniMax H3 Community License Agreement | Same regional terms as above. |
| `moss_image_sft_bf16`, `qwen3.5_9b_bf16`, `flux2-vae_small` (Comfy-Org/MOSS-Image) | Apache-2.0 | Per the repository metadata at the pinned commit. |
| `moss_community/*` LoRAs | Set by each LoRA author | Manual download; check the author's page. |
| `face_detection_yunet_2023mar.onnx` (opencv/face_detection_yunet) | MIT | Optional QA detector, manual download. Training-data licence not checked. |
| `face_recognition_sface_2021dec.onnx` (opencv/face_recognition_sface) | Apache-2.0 | Optional QA recogniser, manual download. Training-data licence not checked. |

## Licences you may meet when extending the kit

- Qwen Research License: some Qwen-family checkpoints (for example certain larger or
  vision-language variants) are released under the Qwen Research License, which limits
  commercial use. The default recipes do not require such a checkpoint, but check the
  licence of any Qwen weight you substitute.
- FLUX Kommunity License: the gated FLUX.3 Klein repositories on Hugging Face are
  distributed under the FLUX Kommunity License and require you to accept it with your
  own account. This kit never downloads them; see `manual_steps` in `setup/models.yaml`.
- Apache-2.0 and MIT: the remaining open components named above.

Nothing here is legal advice. Licence names are summarised from upstream metadata at the
time of writing; the upstream text always governs.
