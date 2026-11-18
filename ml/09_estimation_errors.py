"""
Quantitative portion-size and nutrient-estimation error (reviewer request).

Two small, honest measurement protocols - nothing here is simulated:

A) PORTION ERROR  -> templates/portion_measurements.csv
   Weigh real servings on a kitchen scale, photograph each one, run it through the app and
   copy the weight the app displayed.   columns: sample_id, food, measured_weight_g, app_weight_g
   (Aim for >= 30 samples over >= 10 dishes.)

B) NUTRIENT-TABLE ERROR -> templates/nutrient_reference.csv
   For each food, copy the PER-100 g values from a published composition table
   (IFCT 2017 for Indian foods, USDA FoodData Central as fallback) and record the source.
   columns: food, ref_kcal, ref_protein_g, ref_carb_g, ref_fat_g, ref_sugar_g, ref_sodium_mg, source

Reports: portion MAE / RMSE / MAPE / bias; per-nutrient database-vs-reference MAE / MAPE;
and an end-to-end calorie error split into  (table error) + (portion error)  components.
Use --update-db to write nutrition_db.updated.csv with reference values replacing the stored ones.

Usage:
    python ml/09_estimation_errors.py --portions templates/portion_measurements.csv \
                                      --reference templates/nutrient_reference.csv [--update-db]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "knowledge" / "nutrition_db.csv"
REPORTS = ROOT / "models" / "reports"
NUTS = {"kcal": "ref_kcal", "protein_g": "ref_protein_g", "carb_g": "ref_carb_g",
        "fat_g": "ref_fat_g", "sugar_g": "ref_sugar_g", "sodium_mg": "ref_sodium_mg"}


def mape(est, true):
    est, true = np.asarray(est, float), np.asarray(true, float)
    m = (true != 0) & np.isfinite(true) & np.isfinite(est)
    return float(np.mean(np.abs(est[m] - true[m]) / true[m]) * 100) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portions", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--update-db", action="store_true")
    args = ap.parse_args()

    db = pd.read_csv(DB).set_index("food")
    por = pd.read_csv(args.portions)
    ref = pd.read_csv(args.reference).set_index("food")
    out = {}

    # ---- A) portion error ----
    e = por["app_weight_g"] - por["measured_weight_g"]
    out["portion"] = {
        "n_samples": int(len(por)), "n_foods": int(por["food"].nunique()),
        "MAE_g": round(float(e.abs().mean()), 2), "RMSE_g": round(float(np.sqrt((e ** 2).mean())), 2),
        "MAPE_pct": round(mape(por["app_weight_g"], por["measured_weight_g"]), 2),
        "mean_signed_bias_g": round(float(e.mean()), 2),
    }

    # ---- B) nutrient table vs reference (per 100 g) ----
    common = [f for f in ref.index if f in db.index]
    out["nutrient_table"] = {"n_foods_compared": len(common)}
    for col, rcol in NUTS.items():
        est, tru = db.loc[common, col].astype(float), ref.loc[common, rcol].astype(float)
        ok = est.notna() & tru.notna()                      # blank reference cells are skipped, not guessed
        out["nutrient_table"][col] = {"n_foods": int(ok.sum()),
                                      "MAE": round(float((est[ok] - tru[ok]).abs().mean()), 2) if ok.any() else None,
                                      "MAPE_pct": round(mape(est[ok], tru[ok]), 2) if ok.any() else None}

    # ---- C) end-to-end calorie error and its decomposition ----
    p = por[por["food"].isin(common)].copy()
    p = p[ref.loc[p["food"], "ref_kcal"].notna().values]
    if len(p):
        db_k, ref_k = db.loc[p["food"], "kcal"].values, ref.loc[p["food"], "ref_kcal"].values
        true_kcal = ref_k * p["measured_weight_g"].values / 100
        total = db_k * p["app_weight_g"].values / 100
        table_only = db_k * p["measured_weight_g"].values / 100      # correct portion, table values
        portion_only = ref_k * p["app_weight_g"].values / 100        # correct table, app portion
        out["end_to_end_kcal"] = {
            "n_samples": int(len(p)),
            "total_MAE_kcal": round(float(np.mean(np.abs(total - true_kcal))), 1),
            "total_MAPE_pct": round(mape(total, true_kcal), 2),
            "table_only_MAE_kcal": round(float(np.mean(np.abs(table_only - true_kcal))), 1),
            "portion_only_MAE_kcal": round(float(np.mean(np.abs(portion_only - true_kcal))), 1),
        }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "estimation_errors.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved -> {REPORTS/'estimation_errors.json'}")

    if args.update_db:
        upd = pd.read_csv(DB)
        for f in common:
            for col, rcol in NUTS.items():
                upd.loc[upd["food"] == f, col] = ref.loc[f, rcol]
        dest = DB.with_name("nutrition_db.updated.csv")
        upd.to_csv(dest, index=False)
        print(f"Wrote {dest} ({len(common)} foods overwritten with reference values). "
              f"Review it, then replace nutrition_db.csv and re-run the tests.")


if __name__ == "__main__":
    main()
