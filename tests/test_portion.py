from backend.services import portion


def test_default_portion_matches_reference_serving():
    assert portion.default_portion("idli") == 90.0


def test_bbox_area_scales_portion_up_and_down():
    small = portion.estimate_from_bbox_area("idli", 0.05)
    normal = portion.estimate_from_bbox_area("idli", 0.25)
    large = portion.estimate_from_bbox_area("idli", 0.6)
    assert small < normal < large
