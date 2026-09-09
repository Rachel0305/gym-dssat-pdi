from __future__ import annotations

import unittest

from controller import ControllerParams, DailyStateFeedbackController


def params(**overrides):
    base = dict(
        n_threshold=0.10,
        n_dose=20,
        n_min_interval_days=3,
        n_season_budget=40,
        n_start_dap=7,
        n_stop_dap=90,
        water_threshold=0.20,
        irrigation_dose=15,
        irrigation_min_interval_days=3,
        irrigation_season_budget=30,
        irrigation_start_dap=7,
        irrigation_stop_dap=90,
    )
    base.update(overrides)
    return ControllerParams(**base)


class ControllerTests(unittest.TestCase):
    def test_water_and_n_are_independent(self):
        ctl = DailyStateFeedbackController(params())
        decision = ctl.decide(7, swfac=0.30, nstres=0.0)
        self.assertEqual(decision.irrigation_action_mm, 15.0)
        self.assertEqual(decision.nitrogen_action_kg_ha, 0.0)

        decision = ctl.decide(10, swfac=0.0, nstres=0.30)
        self.assertEqual(decision.irrigation_action_mm, 0.0)
        self.assertEqual(decision.nitrogen_action_kg_ha, 20.0)

    def test_interval_and_budget(self):
        ctl = DailyStateFeedbackController(params())
        self.assertEqual(ctl.decide(7, 0.3, 0.3).irrigation_action_mm, 15.0)
        self.assertEqual(ctl.decide(8, 0.3, 0.3).irrigation_action_mm, 0.0)
        self.assertEqual(ctl.decide(10, 0.3, 0.3).irrigation_action_mm, 15.0)
        self.assertEqual(ctl.decide(13, 0.3, 0.3).irrigation_action_mm, 0.0)

    def test_window_blocks_action(self):
        ctl = DailyStateFeedbackController(params())
        decision = ctl.decide(6, 1.0, 1.0)
        self.assertEqual(decision.irrigation_action_mm, 0.0)
        self.assertEqual(decision.nitrogen_action_kg_ha, 0.0)

    def test_threshold_change_changes_response(self):
        low = DailyStateFeedbackController(params(n_threshold=0.05))
        high = DailyStateFeedbackController(params(n_threshold=0.20))
        self.assertEqual(low.decide(7, 0.0, 0.10).nitrogen_action_kg_ha, 20.0)
        self.assertEqual(high.decide(7, 0.0, 0.10).nitrogen_action_kg_ha, 0.0)


if __name__ == "__main__":
    unittest.main()
