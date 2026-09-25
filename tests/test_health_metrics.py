from backend.services import health_metrics


def test_bmi_calculation():
    assert health_metrics.compute_bmi(70, 175) == round(70 / (1.75 ** 2), 1)


def test_bmi_category_boundaries():
    assert health_metrics.bmi_category(17) == "underweight"
    assert health_metrics.bmi_category(22) == "normal"
    assert health_metrics.bmi_category(27) == "overweight"
    assert health_metrics.bmi_category(32) == "obese"


def test_kcal_target_positive():
    val = health_metrics.daily_kcal_target(25, "female", 165, 60, "moderate")
    assert val > 1000
