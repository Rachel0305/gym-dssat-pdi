"""Read-only WP_ET/PFP_N replay for YC seed2 paired 2K checkpoints."""

from pathlib import Path
import replay_178_multisite_wp_et as replay

ROOT = Path(__file__).resolve().parents[1]
replay.OUT = ROOT / "benchmark_results" / "188YCA_yc_seed2_wp_et_replay"
replay.REPORT_PATH = ROOT / "docs" / "188YCA_yc_seed2_wp_et_replay_record.md"
replay.POLICIES = {
    "yca_forecast_2k_seed2": {
        "station": "YCA", "site": "YC", "seed": 2, "checkpoint_step": 2000,
        "kind": "forecast", "cfg": ROOT / "configs/186YCA_yca_lowIC_dual_branch_weather_reader_2k_seed2.json",
        "inventory": ROOT / "benchmark_results/186YCA_yca_lowIC_dual_branch_weather_reader_2k_seed2_smoke2k/evaluation/186YCA_checkpoint_validation_summary.csv",
    },
    "yca_noforecast_2k_seed2": {
        "station": "YCA", "site": "YC", "seed": 2, "checkpoint_step": 2000,
        "kind": "noforecast", "cfg": ROOT / "configs/187YCA_yca_lowIC_dual_branch_noforecast_control_2k_seed2.json",
        "inventory": ROOT / "benchmark_results/187YCA_yca_lowIC_dual_branch_noforecast_control_2k_seed2_smoke2k/evaluation/187YCA_checkpoint_validation_summary.csv",
    },
}

if __name__ == "__main__":
    print(replay.run())
