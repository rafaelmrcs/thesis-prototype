import unittest
from unittest.mock import patch

try:
    from src import api
except ModuleNotFoundError as exc:
    api = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def node(lat: float, lon: float) -> dict[str, float]:
    return {"lat": lat, "lon": lon}


class DummyModel:
    def predict(self, X):
        return [api.KWH_TO_J * 5.0]


@unittest.skipIf(api is None, f"backend dependencies unavailable: {IMPORT_ERROR}")
class ExactRooftopLookupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.square = [
            node(0.0, 0.0),
            node(0.0, 1.0),
            node(1.0, 1.0),
            node(1.0, 0.0),
        ]

    def test_point_inside_polygon(self) -> None:
        self.assertTrue(api._point_in_polygon(0.5, 0.5, self.square))

    def test_point_outside_polygon(self) -> None:
        self.assertFalse(api._point_in_polygon(1.5, 0.5, self.square))

    def test_point_on_polygon_edge_counts_as_inside(self) -> None:
        self.assertTrue(api._point_in_polygon(0.0, 0.5, self.square))

    def test_smallest_containing_building_wins_when_polygons_overlap(self) -> None:
        big = [
            node(0.0, 0.0),
            node(0.0, 2.0),
            node(2.0, 2.0),
            node(2.0, 0.0),
        ]
        small = [
            node(0.25, 0.25),
            node(0.25, 0.75),
            node(0.75, 0.75),
            node(0.75, 0.25),
        ]
        elements = [{"geometry": big}, {"geometry": small}]

        self.assertEqual(api._containing_building_index(0.5, 0.5, elements), 1)

    def test_no_containing_building_raises_404(self) -> None:
        with self.assertRaises(api.LiveFeatureError) as ctx:
            api._containing_building_index(1.5, 0.5, [{"geometry": self.square}])

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("No mapped rooftop contains", ctx.exception.detail)

    def test_live_feature_bundle_validates_rooftop_before_weather_calls(self) -> None:
        with patch.object(
            api,
            "compute_osm_features",
            side_effect=api.LiveFeatureError("No mapped rooftop contains the selected point.", 404),
        ), patch.object(api, "fetch_nasa_live") as nasa:
            with self.assertRaises(api.LiveFeatureError):
                api._live_feature_bundle(0.5, 0.5)

        nasa.assert_not_called()

    def test_rooftop_area_validation_calculates_error_fields(self) -> None:
        validation = api._compute_rooftop_area_validation(120.0, 100.0)

        self.assertIsNotNone(validation)
        self.assertEqual(validation.predictedArea, 120.0)
        self.assertEqual(validation.actualArea, 100.0)
        self.assertEqual(validation.absoluteError, 20.0)
        self.assertEqual(validation.percentError, 20.0)
        self.assertEqual(validation.squaredError, 400.0)

    def test_predict_omits_rooftop_validation_without_actual_area(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError as exc:
            self.skipTest(f"FastAPI test client unavailable: {exc}")

        client = TestClient(api.app)
        with patch.object(api.ctx, "ensure_core_loaded"), patch.object(
            api.ctx,
            "fi_model",
            DummyModel(),
        ), patch.object(api, "_live_feature_bundle", return_value=self._mock_bundle()):
            response = client.post("/predict", json={"lat": 0.5, "lng": 0.5})

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("rooftopAreaValidation", response.json())

    def test_predict_includes_rooftop_validation_with_actual_area(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ModuleNotFoundError as exc:
            self.skipTest(f"FastAPI test client unavailable: {exc}")

        client = TestClient(api.app)
        with patch.object(api.ctx, "ensure_core_loaded"), patch.object(
            api.ctx,
            "fi_model",
            DummyModel(),
        ), patch.object(api, "_live_feature_bundle", return_value=self._mock_bundle()):
            response = client.post(
                "/predict",
                json={"lat": 0.5, "lng": 0.5, "actualRooftopArea": 80.0},
            )

        self.assertEqual(response.status_code, 200)
        validation = response.json()["rooftopAreaValidation"]
        self.assertEqual(validation["predictedArea"], 100.0)
        self.assertEqual(validation["actualArea"], 80.0)
        self.assertEqual(validation["absoluteError"], 20.0)
        self.assertEqual(validation["percentError"], 25.0)
        self.assertEqual(validation["squaredError"], 400.0)

    @staticmethod
    def _mock_bundle():
        osm = {
            "rooftop_area_sq_m": 100.0,
            "azimuth_deg": 180.0,
            "orientation_score": 1.0,
            "shading_factor": 0.0,
            "SEI_norm": 0.5,
        }
        nasa = {
            "kt": 0.5,
            "ghi_kwh": 4.0,
            "temp_c": 30.0,
            "humidity_pct": 75.0,
        }
        pvlib_features = {
            "ghi_clear_annual": 5.0,
            "sunshine_hours": 6.0,
        }
        return osm, nasa, pvlib_features


if __name__ == "__main__":
    unittest.main()
