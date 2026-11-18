"""
Expert validation workflow for the disease-suitability rules (reviewer request).

  export : writes an Excel workbook for a qualified dietitian/physician containing
             - Rules          : every threshold, flag list and advice string, with columns to agree / correct
             - Verdict audit  : for every food in the database, the system's ok/caution/avoid verdict for each of
                                the six conditions at the default serving, plus EMPTY columns where the expert
                                enters their own verdict (drop-down: ok / caution / avoid)
             - Swap audit     : every suggested substitution, with an accept / reject column
  score  : reads the completed workbook and reports system-vs-expert agreement
             (overall %, per condition %, Cohen's kappa, disagreement list) and swap acceptance rate.

Usage:
    python ml/10_expert_review.py export --out expert_review_sheet.xlsx
    python ml/10_expert_review.py score  --file expert_review_sheet_completed.xlsx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from backend.services import nutrition, recommender, rules_engine

CONDS = ["diabetes", "hypertension", "obesity", "heart_disease", "kidney_disease", "pcos"]
HDR = PatternFill("solid", start_color="2D2D86", end_color="2D2D86")
INPUT = PatternFill("solid", start_color="FFF2CC", end_color="FFF2CC")


def _header(ws, row, titles):
    for c, t in enumerate(titles, 1):
        cell = ws.cell(row=row, column=c, value=t)
        cell.font = Font(bold=True, color="FFFFFF"); cell.fill = HDR
        cell.alignment = Alignment(wrap_text=True, vertical="center")


def export(out: Path):
    wb = Workbook()
    ws = wb.active; ws.title = "Instructions"
    for i, line in enumerate([
        "Expert review of the disease-aware dietary rules (food image recognition project)",
        "",
        "1. 'Rules' sheet: for every row mark Agree = Y/N and, if N, enter your recommended value or comment.",
        "2. 'Verdict audit' sheet: the system's verdict for each food at its default serving is shown for six conditions.",
        "   Enter YOUR verdict in the yellow columns (ok / caution / avoid). Leave blank if you cannot judge.",
        "3. 'Swap audit' sheet: mark each suggested substitution Y (clinically acceptable) or N.",
        "4. Values in the database are per 100 g; 'serving' is the default portion shown to the user.",
        "5. Please record your name, qualification and registration number below.",
        "",
        "Reviewer name:", "Qualification / registration no.:", "Date:"], 1):
        ws.cell(row=i, column=1, value=line)
    ws["A1"].font = Font(bold=True, size=13); ws.column_dimensions["A"].width = 110

    # ---- Rules ----
    rules = json.loads((ROOT / "data/knowledge/disease_rules.json").read_text())
    ws = wb.create_sheet("Rules")
    _header(ws, 1, ["Condition", "Type", "Item", "System value", "Expert agrees (Y/N)", "Expert value / comment"])
    r = 2
    for cond, rule in rules.items():
        if cond.startswith("_"):
            continue
        rows = [("per-serving limit", n, v) for n, v in rule.get("item_limits", {}).items()]
        rows += [("per-meal limit", n, v) for n, v in rule.get("meal_limits", {}).items()]
        rows += [("avoid GI", "glycaemic index", ", ".join(rule.get("avoid_gi", [])) or "-"),
                 ("avoid tags", "food tags", ", ".join(rule.get("avoid_tags", [])) or "-"),
                 ("prefer tags", "food tags", ", ".join(rule.get("prefer_tags", [])) or "-")]
        rows += [("advice", "text", a) for a in rule.get("advice", [])]
        for typ, item, val in rows:
            for c, v in enumerate([cond, typ, item, val], 1):
                ws.cell(row=r, column=c, value=v).alignment = Alignment(wrap_text=True, vertical="top")
            for c in (5, 6):
                ws.cell(row=r, column=c).fill = INPUT
            r += 1
    for col, w in zip("ABCDEF", (16, 18, 24, 60, 18, 50)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"

    # ---- Verdict audit ----
    ws = wb.create_sheet("Verdict audit")
    base = ["food", "serving_g", "kcal", "protein_g", "carb_g", "sugar_g", "fat_g", "sat_fat_g", "sodium_mg", "GI"]
    _header(ws, 1, base + [f"System: {c}" for c in CONDS] + [f"EXPERT: {c}" for c in CONDS])
    dv = DataValidation(type="list", formula1='"ok,caution,avoid"', allow_blank=True)
    ws.add_data_validation(dv)
    for i, food in enumerate(sorted(nutrition.all_foods()), 2):
        row = nutrition.get_food_row(food)
        p = nutrition.scale_to_portion(food, row["serving_g"])
        vals = [food, row["serving_g"], p["kcal"], p["protein_g"], p["carb_g"], p["sugar_g"], p["fat_g"],
                p["sat_fat_g"], p["sodium_mg"], nutrition.food_gi(food)]
        flags = rules_engine.evaluate_food(food, p, CONDS)
        vals += [flags[c]["verdict"] for c in CONDS]
        for c, v in enumerate(vals, 1):
            ws.cell(row=i, column=c, value=v)
        for k in range(len(CONDS)):
            cell = ws.cell(row=i, column=len(base) + len(CONDS) + 1 + k); cell.fill = INPUT; dv.add(cell)
    ws.freeze_panes = "B2"; ws.column_dimensions["A"].width = 24
    for c in range(2, len(base) + 2 * len(CONDS) + 1):
        ws.column_dimensions[ws.cell(row=1, column=c).column_letter].width = 14

    # ---- Swap audit ----
    ws = wb.create_sheet("Swap audit")
    _header(ws, 1, ["Original food", "Suggested alternative", "Expert: acceptable (Y/N)", "Comment"])
    dv2 = DataValidation(type="list", formula1='"Y,N"', allow_blank=True); ws.add_data_validation(dv2)
    r = 2
    for orig, alts in recommender.load_swaps().items():
        if orig.startswith("_"):
            continue
        for a in alts:
            ws.cell(row=r, column=1, value=orig); ws.cell(row=r, column=2, value=a)
            for c in (3, 4):
                ws.cell(row=r, column=c).fill = INPUT
            dv2.add(ws.cell(row=r, column=3)); r += 1
    for col, w in zip("ABCD", (24, 24, 24, 50)):
        ws.column_dimensions[col].width = w
    wb.save(out)
    print(f"Wrote {out}  (hand this to the dietitian; then run 'score' on the completed file)")


def score(path: Path):
    from sklearn.metrics import cohen_kappa_score
    df = pd.read_excel(path, sheet_name="Verdict audit")
    res = {"per_condition": {}}
    sys_all, exp_all, dis = [], [], []
    for c in CONDS:
        s, e = df[f"System: {c}"], df[f"EXPERT: {c}"]
        m = e.notna()
        if m.sum() == 0:
            continue
        s_, e_ = s[m].astype(str).str.lower(), e[m].astype(str).str.strip().str.lower()
        res["per_condition"][c] = {"n": int(m.sum()), "agreement_pct": round(float((s_ == e_).mean() * 100), 1)}
        sys_all += list(s_); exp_all += list(e_)
        for f, a, b in zip(df.loc[m, "food"], s_, e_):
            if a != b:
                dis.append({"food": f, "condition": c, "system": a, "expert": b})
    if sys_all:
        res["overall_agreement_pct"] = round(100 * sum(a == b for a, b in zip(sys_all, exp_all)) / len(sys_all), 1)
        res["cohen_kappa"] = round(float(cohen_kappa_score(sys_all, exp_all)), 3)
        res["n_judgements"] = len(sys_all)
        # 'unsafe disagreement': system says ok while expert says avoid
        res["unsafe_disagreements(system ok, expert avoid)"] = sum(d["system"] == "ok" and d["expert"] == "avoid" for d in dis)
    sw = pd.read_excel(path, sheet_name="Swap audit")
    ans = sw["Expert: acceptable (Y/N)"].dropna().astype(str).str.upper()
    if len(ans):
        res["swap_acceptance_pct"] = round(float((ans == "Y").mean() * 100), 1); res["swaps_reviewed"] = int(len(ans))
    out = ROOT / "models" / "reports"; out.mkdir(parents=True, exist_ok=True)
    (out / "expert_agreement.json").write_text(json.dumps({**res, "disagreements": dis}, indent=2))
    pd.DataFrame(dis).to_csv(out / "expert_disagreements.csv", index=False)
    print(json.dumps(res, indent=2)); print(f"Disagreement list -> {out/'expert_disagreements.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--out", default="expert_review_sheet.xlsx")
    s = sub.add_parser("score"); s.add_argument("--file", required=True)
    a = ap.parse_args()
    export(Path(a.out)) if a.cmd == "export" else score(Path(a.file))
