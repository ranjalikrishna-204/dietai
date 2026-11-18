# Camera-ready toolkit (paper 462) - quick map

Files (copy into your project, same paths, overwrite when asked):
  ml/01_prepare_classification_dataset.py  NEW: deterministic split + "Split fingerprint"
  ml/02_train_classifier.py                NEW: more architectures, --ckpt-name, exact head freezing
  ml/07_dedup_check.py                     duplicate / leakage audit
  ml/08_compare_models.py                  train + evaluate several architectures, --collect to merge
  ml/09_estimation_errors.py               portion + nutrient error
  ml/10_expert_review.py                   export / score the dietitian workbook
  backend/services/nutrition.py            FIX: DB values are per 100 g
  tests/test_nutrition.py                  adds a per-100 g test
  templates/*.csv                          empty templates for your measurements
  expert_review_sheet.xlsx                 REGENERATE after the nutrient fix (see guide)
  latex_camera_ready_snippets.tex          paste-ready paper text with \todo{} blanks

Order: nutrition fix -> duplicate audit -> rebuild split -> model comparison (one model per
session) -> evaluate/export the chosen model -> portion/nutrient error -> expert review -> paper.
