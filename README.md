# AI-Based Health Condition-Aware Dietary Recommendation System

A working, end-to-end implementation of the final-year project: upload a food photo, get the
food identified, its nutrition estimated, its suitability checked against your health
conditions (diabetes, hypertension, obesity, heart disease, kidney disease, PCOS), healthier
alternatives suggested, cooking guidance, and a running dietary-analytics dashboard.

> **Disclaimer:** the disease-suitability rules in `data/knowledge/disease_rules.json` are a
> simplified, editable rule engine inspired by general public dietary guidance. This is a
> student project / decision-support demo, **not medical advice** - always defer to a doctor or
> registered dietitian for real dietary decisions, especially for kidney disease where limits
> depend heavily on disease stage.

---

## 1. Which platform should I use?

**Short answer: use VS Code for writing/running the project day-to-day, and a free-GPU
notebook (Google Colab or Kaggle Notebooks) for the actual model training steps.**

| Task | Recommended tool | Why |
|---|---|---|
| Writing backend/frontend code, running FastAPI + Streamlit, debugging, git | **VS Code** | Mature Python/Jupyter extensions, integrated terminal, debugger, huge community support, free and stable. This is what the whole project below assumes. |
| Training the classifier / detector (needs a GPU for reasonable speed) | **Google Colab** (free T4 GPU) or **Kaggle Notebooks** (free P100/T4 GPU, 30 hrs/week) | Your own laptop almost certainly has no/weak GPU; training EfficientNetV2/YOLO on CPU can take many hours per epoch. Colab/Kaggle give free GPU time - open `ml/02_train_classifier.py` there, or just `!python ml/02_train_classifier.py ...`. |
| **Google Antigravity** (Google's new agentic, VS-Code-fork IDE, public preview since late 2025) | Optional | It's a legitimate, free VS Code fork with built-in AI agents (Gemini 3, plus Claude/GPT access) that can autonomously write/run/verify code and even drive a browser to test your Streamlit UI. It is *not required* for this project - everything here is plain Python/FastAPI/Streamlit that runs the same in any editor - but if you want an AI pair-programmer that can execute multi-step tasks (e.g. "add a new disease rule and write its tests") on your behalf, it's worth trying, since all your VS Code extensions/keybindings carry over. Treat it as an alternative front-end to VS Code, not a replacement for the Colab/Kaggle GPU step above (it does not give you a free GPU for training). |

So: **VS Code + Colab/Kaggle for training** is the safe, standard combination for a B.Tech
project. Antigravity is a fine *optional* upgrade to VS Code if you want AI-agent assistance
while coding, but changes nothing about the architecture below.

---

## 2. Model choices (and why they differ slightly from the proposal)

| Proposal said | This implementation uses | Why |
|---|---|---|
| YOLOv12 for food recognition | **EfficientNetV2-S image classifier** (primary) + optional **YOLO26** detector for multi-item plates | Almost every food photo has one dominant dish - classification needs far less labelled data than object detection (just folders of images, no bounding boxes) and reaches higher accuracy for this case. YOLOv12 (and YOLO26) are kept for the optional stretch goal of localizing **multiple** items on one plate/thali photo, where boxes also drive portion-size estimation. |
| — | **YOLO26** instead of YOLOv12 for that optional detector | YOLO26 (Ultralytics, Jan 2026) is the current flagship: natively NMS-free (lower/steadier latency per photo), drops Distribution Focal Loss for cleaner ONNX export, and has the best small-object accuracy in the family. YOLO11 (2024) is suggested as a well-supported fallback if `yolo26n.pt` isn't available in your installed Ultralytics version. |
| Grad-CAM/SHAP explainability | Not implemented in v1 (flagged as a clear "next step") | Kept the codebase focused; `ml/03_evaluate_classifier.py` already gives a confusion matrix + per-class report, which is the most useful diagnostic for a first version. Grad-CAM can be bolted onto the trained classifier later (`pip install grad-cam`). |

---

## 3. Full file structure

```
dietai/
├── README.md                     <- this file
├── requirements.txt               <- runtime deps (backend + frontend + tests)
├── requirements-train.txt         <- + training-only deps (torch, torchvision, ultralytics...)
├── config.py                      <- ALL tunable settings in one place (paths, thresholds)
├── .env.example                   <- copy to .env to override config.py without editing code
├── .gitignore
├── run_backend.sh                 <- `bash run_backend.sh`  -> starts FastAPI on :8000
├── run_frontend.sh                <- `bash run_frontend.sh` -> starts Streamlit on :8501
│
├── data/
│   ├── knowledge/                 <- hand-curated "knowledge base" the whole app runs on
│   │   ├── nutrition_db.csv       <- per-100g/serving nutrition for every recognised food
│   │   ├── disease_rules.json     <- editable thresholds per condition (diabetes, BP, ...)
│   │   ├── healthy_swaps.json     <- food -> healthier alternative(s) graph
│   │   ├── cooking_tips.json      <- tag/food/condition -> cooking guidance strings
│   │   └── class_map.json         <- raw dataset label -> canonical class name aliases
│   ├── raw/                       <- YOU download datasets here (gitignored, see ml/01_...)
│   └── processed/                 <- auto-generated train/val/test splits (gitignored)
│
├── ml/                             <- dataset prep, training, evaluation, export (run these
│   │                                  in order once, on Colab/Kaggle for the GPU steps)
│   ├── class_map.py                <- shared label-normalisation helper
│   ├── 01_prepare_classification_dataset.py   <- raw images -> clean train/val/test split
│   ├── 02_train_classifier.py                 <- fine-tune EfficientNetV2-S / MobileNetV3
│   ├── 03_evaluate_classifier.py               <- accuracy, top-5, confusion matrix, report
│   ├── 04_export_model.py                      <- .pt -> models/classifier.onnx (+ metadata)
│   ├── 05_prepare_detection_dataset.py [optional]  <- annotated boxes -> YOLO dataset format
│   └── 06_train_detector.py         [optional]  <- fine-tune YOLO26/YOLO11 for multi-item plates
│
├── models/                         <- trained artifacts land here (gitignored)
│   ├── classifier_best.pt          <- produced by 02_train_classifier.py
│   ├── classifier.onnx             <- produced by 04_export_model.py  <- what the backend loads
│   ├── classifier.json             <- classes + preprocessing metadata for the ONNX model
│   ├── detector_best.pt            <- [optional] produced by 06_train_detector.py
│   └── reports/                    <- confusion_matrix.png, classification_report.json
│
├── backend/                         <- FastAPI application (the "server")
│   ├── main.py                      <- all HTTP endpoints
│   ├── schemas.py                   <- Pydantic request/response models
│   └── services/                    <- the actual logic, one responsibility per file
│       ├── preprocessing.py         <- OpenCV image decode/resize/(optional) denoise/CLAHE
│       ├── recognizer.py            <- loads ONNX classifier + optional YOLO detector
│       ├── portion.py               <- bbox-area -> gram heuristic
│       ├── nutrition.py             <- reads nutrition_db.csv, scales to portion size
│       ├── rules_engine.py          <- disease_rules.json -> per-food/per-meal suitability
│       ├── recommender.py           <- healthy_swaps.json + cooking_tips.json -> suggestions
│       ├── health_metrics.py        <- BMI + daily calorie target (Mifflin-St Jeor)
│       └── database.py              <- SQLite: user profiles + meal logs + analytics queries
│
├── frontend/
│   └── app.py                       <- Streamlit dashboard (upload photo, see results, charts)
│
└── tests/                           <- pytest suite (24 tests, all passing) covering every
    ├── conftest.py                     service module above + full API integration tests
    ├── test_nutrition.py
    ├── test_rules_engine.py
    ├── test_recommender.py
    ├── test_portion.py
    ├── test_health_metrics.py
    └── test_api.py
```

---

## 4. Setup - from zero to running app

### 4.1 Environment

```bash
git clone <your-repo-url> dietai   # or just open this folder in VS Code
cd dietai
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # backend + frontend + tests (no torch/ultralytics needed yet)
cp .env.example .env               # then edit .env if you want different ports/paths
```

Open the folder in VS Code (`code .`). Recommended extensions: **Python**, **Pylance**,
**Jupyter** (for exploring datasets), **SQLite Viewer** (to inspect `dietai.db`).

### 4.2 Get a dataset (Step 1 of the ML pipeline)

Pick at least one (more classes/images = a better model):

* **Kaggle - "Indian Food Images Dataset"** (`iamsouravbanerjee/indian-food-images-dataset`,
  ~4,000 images / 80 classes):
  ```bash
  pip install kaggle          # needs a kaggle.json API token in ~/.kaggle/
  kaggle datasets download -d iamsouravbanerjee/indian-food-images-dataset -p data/raw/indian --unzip
  ```
* **Kaggle - Food-101** (`kmader/food41`, 101,000 images / 101 global classes - good source for
  pizza/burger/fries/ice-cream/omelette classes that aren't in Indian-only datasets):
  ```bash
  kaggle datasets download -d kmader/food41 -p data/raw/food101 --unzip
  ```
* **Hugging Face** (smaller, good for a first smoke-test):
  ```python
  from datasets import load_dataset
  ds = load_dataset("rajistics/indian_food_images")
  # then save ds["train"] images out to data/raw/hf_indian/<label>/*.jpg
  ```

Whatever you use, you need the layout `data/raw/<source_name>/<class_folder>/*.jpg`.

```bash
pip install -r requirements-train.txt     # adds torch, torchvision, ultralytics, datasets...
python ml/01_prepare_classification_dataset.py --list-labels   # sanity check: see what labels you got
# add any missing aliases to data/knowledge/class_map.json, then:
python ml/01_prepare_classification_dataset.py
```
This writes `data/processed/classification/{train,val,test}/<class>/*.jpg`.

### 4.3 Train (Step 2) - do this on Colab or Kaggle Notebooks (free GPU)

Upload the `data/processed/classification/` folder (or mount Google Drive / use the Kaggle
dataset uploader), then:
```bash
python ml/02_train_classifier.py --arch efficientnet_v2_s --epochs 15 --batch-size 32
```
Swap `--arch mobilenet_v3_small` for a much smaller/faster model if you also want to try
on-device/edge inference later. Best checkpoint -> `models/classifier_best.pt`.

### 4.4 Evaluate (Step 3)
```bash
python ml/03_evaluate_classifier.py --checkpoint models/classifier_best.pt
```
-> `models/reports/classification_report.json` and `confusion_matrix.png`.

### 4.5 Export for serving (Step 4)
```bash
python ml/04_export_model.py --checkpoint models/classifier_best.pt --out models/classifier.onnx
```
Download `models/classifier.onnx` + `models/classifier.json` back to your local machine if you
trained on Colab/Kaggle.

### 4.6 (Optional) multi-item detector - Steps 5 & 6
See the docstrings at the top of `ml/05_prepare_detection_dataset.py` and
`ml/06_train_detector.py` for where to get an annotated (bounding-box) dataset and how to train
YOLO26/YOLO11 on it. Only needed if you want per-item boxes on crowded thali photos; the app
works fully without it (falls back to the classifier).

### 4.7 Run the app
```bash
bash run_backend.sh     # terminal 1 -> http://127.0.0.1:8000/docs (interactive API docs)
bash run_frontend.sh    # terminal 2 -> http://localhost:8501
```
Create a profile in the sidebar, upload a food photo, and you'll get recognition, nutrition,
disease-suitability flags, healthier alternatives, cooking tips, and a logging/analytics tab.

### 4.8 Run the tests
```bash
pytest tests/ -v
```
All service logic (nutrition math, rule engine, recommender, portion heuristic, BMI/calorie
math) and the full API are covered - 24 tests. `test_api.py`'s auto-recognition test is
automatically skipped until `models/classifier.onnx` exists (it doesn't require a trained
model to test everything else, including the manual-override path).

---

## 5. Extending the knowledge base

Everything domain-specific lives in `data/knowledge/*` as plain CSV/JSON - no code changes
needed to:
* **Add a food:** append a row to `nutrition_db.csv` (and add its label + aliases to
  `class_map.json` so the recognizer can map raw dataset labels onto it).
* **Add/adjust a disease rule:** edit the relevant block in `disease_rules.json` (thresholds
  and advice strings are all editable, and should be reviewed against current dietitian
  guidance before real use).
* **Add a healthier swap:** add an entry to `healthy_swaps.json` (`food: [alternative, ...]`) -
  it's automatically checked against the user's conditions/allergies/diet before being shown.
* **Add a cooking tip:** add to `cooking_tips.json` under `by_tag`, `by_food`, or `by_condition`.

## 6. Roadmap for the remaining project scope

- [ ] Grad-CAM/SHAP visual explanations on top of the trained classifier
- [ ] Train and wire in the optional YOLO26 multi-item detector for thali photos
- [ ] Portion estimation via a reference object (plate/coin) instead of the area heuristic
- [ ] User authentication (the current `/users` endpoint has no login/password - fine for a
      demo, not for a real deployment)
- [ ] Deploy: containerize backend (Dockerfile) + host Streamlit (Streamlit Community Cloud /
      HF Spaces) or replace with a React frontend calling the same FastAPI backend
