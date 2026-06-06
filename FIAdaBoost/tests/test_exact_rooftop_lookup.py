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


if __name__ == "__main__":
    unittest.main()
