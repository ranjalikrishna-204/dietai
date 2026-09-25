from backend.services import recommender


def test_jalebi_has_fruit_alternative():
    alts = recommender.suggest_alternatives("jalebi", ["diabetes"], "none", [], 60)
    assert any(a["food"] == "fruit_bowl" for a in alts)


def test_diet_preference_filters_non_veg():
    alts = recommender.suggest_alternatives("biryani", [], "vegetarian", [], 250)
    for a in alts:
        assert a["food"] != "grilled_chicken"


def test_allergy_filters_dairy():
    alts = recommender.suggest_alternatives("jalebi", [], "none", ["dairy"], 60)
    for a in alts:
        assert a["food"] != "curd"


def test_cooking_tips_returns_something_for_fried_food():
    tips = recommender.cooking_tips_for("samosa", ["diabetes"])
    assert len(tips) > 0
