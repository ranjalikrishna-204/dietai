from backend.services import nutrition, rules_engine


def test_high_sugar_dessert_flagged_for_diabetes():
    portion = nutrition.scale_to_portion("jalebi", 60)
    flags = rules_engine.evaluate_food("jalebi", portion, ["diabetes"])
    assert flags["diabetes"]["verdict"] in ("caution", "avoid")


def test_lean_protein_ok_for_heart_disease():
    portion = nutrition.scale_to_portion("grilled_chicken", 150)
    flags = rules_engine.evaluate_food("grilled_chicken", portion, ["heart_disease"])
    assert flags["heart_disease"]["verdict"] == "ok"


def test_high_sodium_flagged_for_hypertension():
    portion = nutrition.scale_to_portion("pav_bhaji", 250)
    flags = rules_engine.evaluate_food("pav_bhaji", portion, ["hypertension"])
    assert flags["hypertension"]["verdict"] in ("caution", "avoid")


def test_no_conditions_falls_back_to_general():
    portion = nutrition.scale_to_portion("idli", 90)
    flags = rules_engine.evaluate_food("idli", portion, [])
    assert "general" in flags
