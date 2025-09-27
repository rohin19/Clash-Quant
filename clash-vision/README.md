# Clash Vision – Minimal Base (Incremental Implementation)

This folder will be built up step‑by‑step. We are starting from the smallest working piece: "Can we load a YOLOv8 weight file (.pt) and run a single image inference?" Everything else (dataset validation, resizing, training, live screen capture) will be layered on after each milestone is confirmed.

---
## Current Scope (Milestone 1)
Goal: Local inference on one image using a provided Roboflow‑trained YOLOv8 weights file (e.g., `best.pt`).

You should have:
```
clash-vision/
   models/              # (create this) place best.pt here
   src/clash_vision/
      inference/predict_image.py  # current minimal tool we care about
      utils/logger.py             # colored logging helper
```

Optional (present but NOT in scope yet):
- `live_inference.py` (will become Milestone 2)
- `dataset/verify_dataset.py` (later when validating data)
- `training/train.py` (later if we locally fine‑tune)

---
## Setup (Windows PowerShell)
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Place your downloaded weights file here:
```
clash-vision/models/best.pt
```

---
## Run Single Image Inference
```powershell
python -m clash_vision.inference.predict_image --weights models/best.pt --source path\to\image.jpg --show --save
```
Flags:
- `--show` opens a window (press any key to close)
- `--save` writes an annotated `image.pred.jpg`
- `--json` prints structured JSON detections

Expected success criteria:
1. Script runs without import errors.
2. Boxes appear on objects that match trained classes.
3. Confidence values look plausible (0.2–0.9 typical in early models).

If it fails to load the model: verify the path to `best.pt` and that the file is not corrupted.

---
## Next Milestones (Do NOT implement yet)
We will add these one at a time after confirming Milestone 1.
1. Live screen capture (`live_inference.py`).
2. Batch directory inference helper (simple wrapper).
3. Dataset structural validator (`verify_dataset.py`).
4. Optional local fine‑tuning (`training/train.py`).
5. Performance / export utilities (ONNX, etc.).

See `PHASES.md` (will list detailed acceptance criteria as we advance).

---
## Keep It Clean
If you are not using a script for the current milestone, ignore it—no need to modify or run.

---
## Troubleshooting (Restricted to Milestone 1)
- OpenCV window not showing: ensure you’re not in a headless environment.
- CUDA warnings: safe to ignore if you’re on CPU (YOLO falls back automatically).
- No detections: try a different test image known to contain target objects.

---
## License
Internal MVP usage only.

