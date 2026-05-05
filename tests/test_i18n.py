"""Tests for Azerbaijani language pack."""

from __future__ import annotations

import pytest

from visionvoiceasist.i18n import (
    AZ_LABELS,
    EMERGENCY_LABELS,
    SURFACES,
    Messages,
    az_label,
    distance_label,
    position_label,
)


class TestAzLabel:
    def test_known_label(self) -> None:
        assert az_label("car") == "maşın"
        assert az_label("person") == "insan"
        assert az_label("dog") == "it"

    def test_unknown_label_returns_english(self) -> None:
        assert az_label("alien_object") == "alien_object"

    def test_all_80_coco_classes_present(self) -> None:
        assert len(AZ_LABELS) >= 80

    def test_no_empty_translations(self) -> None:
        for eng, az in AZ_LABELS.items():
            assert az.strip(), f"Empty translation for '{eng}'"


class TestPositionLabel:
    def test_left_region(self) -> None:
        # cx at 100, frame_w 640 → left third (0–213)
        label = position_label(100, 640)
        assert label == "Solunuzda"

    def test_right_region(self) -> None:
        # cx at 500 > 640*2/3 = 426
        label = position_label(500, 640)
        assert label == "Sağınızda"

    def test_center_region(self) -> None:
        label = position_label(320, 640)
        assert label == "Qarşınızda"

    def test_diacritics_correct(self) -> None:
        """Regression test — old code had 'Saginizdə' / 'Qarsinizda'."""
        right = position_label(600, 640)
        assert "ğ" in right, f"Missing ğ in '{right}'"
        center = position_label(320, 640)
        assert "ş" in center, f"Missing ş in '{center}'"

    def test_boundary_exactly_left_third(self) -> None:
        # cx == frame_w // 3 → NOT < threshold, so centre
        label = position_label(640 // 3, 640)
        assert label == "Qarşınızda"

    def test_boundary_exactly_right_two_thirds(self) -> None:
        # cx == 2 * frame_w // 3 → NOT > threshold, so centre
        label = position_label(2 * 640 // 3, 640)
        assert label == "Qarşınızda"


class TestDistanceLabel:
    def test_very_close(self) -> None:
        assert "< 1 m" in distance_label(0.6)

    def test_close(self) -> None:
        assert "1–2 m" in distance_label(0.3)

    def test_medium(self) -> None:
        assert "2–4 m" in distance_label(0.1)

    def test_far(self) -> None:
        assert "4+" in distance_label(0.01)

    def test_boundary_critical(self) -> None:
        # threshold is > 0.45, so 0.46 is "very close" but 0.45 is "close"
        assert "< 1 m" in distance_label(0.46)
        assert "1–2 m" in distance_label(0.45)

    def test_boundary_zero(self) -> None:
        assert "4+" in distance_label(0.0)


class TestMessages:
    def test_greeting_battery_format(self) -> None:
        msg = Messages.GREETING_BATTERY.format(pct=45)
        assert "45" in msg

    def test_approach_format(self) -> None:
        msg = Messages.APPROACH.format(label="maşın")
        assert "maşın" in msg

    def test_critical_proximity_format(self) -> None:
        msg = Messages.CRITICAL_PROXIMITY.format(label="insan")
        assert "insan" in msg

    def test_battery_critical_format(self) -> None:
        msg = Messages.BATTERY_CRITICAL.format(pct=5)
        assert "5" in msg

    def test_scene_on_surface_one_format(self) -> None:
        msg = Messages.SCENE_ON_SURFACE_ONE.format(surface="Masa", item="kitab")
        assert "Masa" in msg
        assert "kitab" in msg

    def test_scene_on_surface_few_format(self) -> None:
        msg = Messages.SCENE_ON_SURFACE_FEW.format(
            surface="Masa", items="kitab, qələm", last="stəkan"
        )
        assert "stəkan" in msg

    def test_scene_crowd_format(self) -> None:
        msg = Messages.SCENE_CROWD.format(n=5)
        assert "5" in msg

    def test_all_templates_have_no_unresolved_braces(self) -> None:
        """Verify all format-string templates are valid Python format strings."""
        import inspect
        attrs = {k: v for k, v in inspect.getmembers(Messages)
                 if not k.startswith("_") and isinstance(v, str)}
        for name, val in attrs.items():
            # Should not contain unmatched { / }
            try:
                # Quick parse — if placeholder fields listed, just format dummy
                import string
                parser = string.Formatter()
                list(parser.parse(val))  # raises if malformed
            except Exception as e:
                pytest.fail(f"Messages.{name} has malformed template: {e}")


class TestEmergencyAndSurfaces:
    def test_person_is_emergency(self) -> None:
        assert "person" in EMERGENCY_LABELS

    def test_car_is_emergency(self) -> None:
        assert "car" in EMERGENCY_LABELS

    def test_table_is_surface(self) -> None:
        assert "dining table" in SURFACES

    def test_person_not_surface(self) -> None:
        assert "person" not in SURFACES
