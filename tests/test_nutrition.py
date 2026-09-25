from backend.services import nutrition


def test_load_table_has_expected_columns():
    df = nutrition.load_table()
    assert "idli" in df.index
    assert "kcal" in df.columns


def test_scale_to_portion_doubles_correctly():
    row = nutrition.get_food_row("idli")
    base = nutrition.scale_to_portion("idli", row["serving_g"])
    doubled = nutrition.scale_to_portion("idli", row["serving_g"] * 2)
    assert doubled["kcal"] == round(base["kcal"] * 2, 2)


def test_unknown_food_raises():
    try:
        nutrition.get_food_row("not_a_real_food")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_sum_meal():
    a = nutrition.scale_to_portion("idli", 90)
    b = nutrition.scale_to_portion("sambar", 150)
    totals = nutrition.sum_meal([a, b])
    assert totals["kcal"] == round(a["kcal"] + b["kcal"], 2)
