from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "225_yc_dqn_readiness_audit"
RESULTS = OUT / "results"
DOCS = OUT / "docs"


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    path = RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def contract_rows() -> list[dict[str, object]]:
    return [
        {"component": "site", "ppo_reference": "YCA/YC", "dqn_candidate": "032_00=LCA; 031_19=SYA; 029=multi-site legacy", "status": "MISMATCH", "evidence": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:415-444; src/run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19.py:404-414", "required_adapter": "Build the DQN on the same YC wrapper and source input as 055_00; do not reuse LCA/SYA smoke artifacts."},
        {"component": "train_validation_years", "ppo_reference": "train 2004-2013; validation 2014-2023", "dqn_candidate": "032_00 single LCA2010; 031_19 single SYA2014; 029 historical YC stage years", "status": "MISMATCH", "evidence": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/055_00_run_manifest.json; experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml", "required_adapter": "Reuse the exact 10+10 YC split and preserve frozen validation-only evaluation."},
        {"component": "input_profile_and_IC", "ppo_reference": "lowIC, YC CNYC0801.MZX, validated preflight", "dqn_candidate": "032 uses all-year calibration root with LCA smoke; 029 uses linked stage inputs", "status": "MISMATCH", "evidence": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/055_00_run_manifest.json; src/run_055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py:80-124", "required_adapter": "Pass the 055_00 lowIC input profile and provenance through the DQN runner."},
        {"component": "cultivar", "ppo_reference": "MZCER048.CUL; ZD0985 Yucheng No985", "dqn_candidate": "032 config points at MZCER048.CUL; 029 stage outputs also carry MZCER048.CUL", "status": "PARTIAL", "evidence": "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml; DSSAT_auto_validation/.../YC/CNYC0801.MZX", "required_adapter": "Verify the resolved YC CUL/MZX pair in the DQN run manifest, not only the filename."},
        {"component": "observation_schema", "ppo_reference": "046_02 raw observation; daily state; no forecast", "dqn_candidate": "032 wrapper preserves base env observation space; 029 is stage obs_dim 25", "status": "PARTIAL", "evidence": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json; src/ppo_evaluate.py:32-42; src/mask_aware_dqn_029.py:100-126", "required_adapter": "Assert observation shape and variable order at reset/step; do not reuse the 25-dim stage checkpoint."},
        {"component": "preprocessing", "ppo_reference": "normalization false; raw values", "dqn_candidate": "032 uses env values; 029 uses float32 arrays", "status": "MATCH_WITH_ASSERTION", "evidence": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json; src/mask_aware_dqn_029.py:137-151", "required_adapter": "Persist an observation schema/hash in the DQN manifest and assert it at load/eval."},
        {"component": "SWFAC_NSTRES_direction", "ppo_reference": "raw SWFAC/NSTRES; stress-aware wrapper uses stress direction", "dqn_candidate": "032 reads/logs same raw state but has different reward; 029 stage has separate reward/scaler", "status": "PARTIAL", "evidence": "src/ppo_evaluate.py:32-42; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:152-169", "required_adapter": "Replay the current 055 reward/safety chain and explicitly audit SWFAC/NSTRES direction."},
        {"component": "reward_formula", "ppo_reference": "040_28 reward + SWFAC guardrail penalty + 040_36 late reserve chain", "dqn_candidate": "032 stress-relief bonus; 031_19 yield minus resource; 029 stage reward", "status": "MISMATCH", "evidence": "src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py:76-100; src/run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py:80-113; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:152-169", "required_adapter": "Use one shared YC reward implementation; do not compare 032 reward against 055 PPO."},
        {"component": "reward_scale", "ppo_reference": "inherited 0.001 scale in current 055 chain", "dqn_candidate": "032/031 use 0.001; 029 stage has its own reward configuration", "status": "PARTIAL", "evidence": "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml; src/run_sya_ppo_configured_046_02.py:158-163", "required_adapter": "Record exact scalar reward and scale in a resolved DQN config."},
        {"component": "action_grid", "ppo_reference": "I=[0,15,30,45] x N=[0,40,80,120] = 16", "dqn_candidate": "032/031 static grid is 16; 029 default is 9", "status": "PARTIAL", "evidence": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:61-64; src/mask_aware_dqn_029.py:104", "required_adapter": "Instantiate custom DQN with action_dim=16 and the exact row-major mapping; assert mapping hash."},
        {"component": "action_mapping", "ppo_reference": "row-major index: i_index*4+n_index", "dqn_candidate": "032 wrapper uses same row-major construction; 029 stage action indices are not the current 16-action map", "status": "PARTIAL", "evidence": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:61-64; src/run_five_site_stage_mask_aware_dqn_029_02.py:118-132", "required_adapter": "Use a single YC mapping object for mask, transition, replay, evaluation, and daily logging."},
        {"component": "daily_caps", "ppo_reference": "I max 45 mm/day; N max 120 kg/ha/day", "dqn_candidate": "032 config same", "status": "MATCH", "evidence": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json; experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml", "required_adapter": "Keep the assertion in the YC DQN preflight."},
        {"component": "seasonal_budget", "ppo_reference": "I=240 mm with staged guardrails; N=250 kg/ha", "dqn_candidate": "032 I=160, N=250; 029 has stage-specific budget/mask", "status": "MISMATCH", "evidence": "src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py:117-125; experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml", "required_adapter": "Set I=240/N=250 and reproduce all staged checks before DQN smoke."},
        {"component": "minimum_intervals", "ppo_reference": "irrigation=7 days; fertilization=7 days", "dqn_candidate": "032 config same; wrapper checks them", "status": "MATCH", "evidence": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:96-120; experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml", "required_adapter": "Verify mask legality after each state update."},
        {"component": "DAP_windows", "ppo_reference": "irrigation [1,120]; fertilization [1,90]", "dqn_candidate": "032 config/base wrapper same", "status": "MATCH", "evidence": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:96-120; experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml", "required_adapter": "Do not allow a separate DQN action-window implementation to drift."},
        {"component": "staged_irrigation_reserve", "ppo_reference": "DAP<=30 I<=75; DAP<=60 I<=150; DAP<=90 I<=195; late reserve 45", "dqn_candidate": "032 only basic I=160 season soft limit; no 040_26/040_36 reserve chain", "status": "MISMATCH", "evidence": "src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_26.py:84-123; src/run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py:80-92", "required_adapter": "Use the current late-reserve wrapper or an equivalent unit-tested, contract-identical adapter."},
        {"component": "SWFAC_guardrail", "ppo_reference": "threshold 0.05; penalty coefficient 50; reward-scale applied", "dqn_candidate": "032 has stress-relief bonus but no current guardrail penalty", "status": "MISMATCH", "evidence": "src/run_sya_lowIC_ppo_i240_swfac_guardrail_reward_040_28.py:76-100; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:152-169", "required_adapter": "Keep identical reward and guardrail behavior for both algorithms."},
        {"component": "termination_and_step_budget", "ppo_reference": "daily DSSAT episode; max_steps=260", "dqn_candidate": "032 base env uses max_steps=260; 029 stage is six decisions/episode", "status": "PARTIAL", "evidence": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:206-208; src/run_five_site_stage_mask_aware_dqn_029_02.py:44-45,118-132", "required_adapter": "Use daily YC episodes and enforce the same termination/output protocol as 055."},
        {"component": "weather_selection", "ppo_reference": "original YC weather; 10 train years + 10 validation years", "dqn_candidate": "032/031 single smoke year; 029 stage historical weather set", "status": "MISMATCH", "evidence": "benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/055_00_run_manifest.json; src/run_five_site_half_split_stress_aware_maskableppo_batch_032_22.py:112-180", "required_adapter": "Wire the exact 055 YC year selection and preserve per-year provenance."},
        {"component": "seed_and_evaluation", "ppo_reference": "seeded PPO checkpoints 25/50/75/100K; deterministic masked evaluation", "dqn_candidate": "032 seed0 final-only 5K; DQN eval masks externally", "status": "MISMATCH", "evidence": "configs/055_00_yca_lowIC_expanded_action_maskableppo.json; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:223-258,333-370", "required_adapter": "Define DQN seeds/checkpoint grid only after adapter smoke; evaluate with internal mask-safe greedy."},
        {"component": "metrics_and_provenance", "ppo_reference": "yield + irrigation + N + PFP_N; WP_ET/NUE only after exact Summary.OUT/ETCP replay", "dqn_candidate": "032 emits daily/eval files but is one LCA year and lacks current YC replay", "status": "PARTIAL", "evidence": "benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke/daily_outputs/LCA/2010_dqn_stress_aware_eval_daily.csv; benchmark_results/055_00_yca_lowIC_expanded_action_maskableppo/055_00_formal_result.json", "required_adapter": "Use the same DSSAT Summary.OUT/ETCP metric replay and retain failed/incomplete outputs."},
        {"component": "checkpoint_and_resume", "ppo_reference": "25/50/75/100K SB3 PPO checkpoints", "dqn_candidate": "032/031 final SB3 zip only; 029 custom checkpoints, but replay buffer is not saved", "status": "MISMATCH", "evidence": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:244-258; src/mask_aware_dqn_029.py:197-228", "required_adapter": "Add an explicit DQN checkpoint schedule; persist replay only if resume semantics are required."},
    ]


def mask_rows() -> list[dict[str, object]]:
    return [
        {"candidate": "032_00_standard_sb3_dqn", "phase": "exploration/warmup", "mask_safe": "NO", "buffer_action_matches_executed_action": "NO_OR_UNSAFE", "evidence": "stable_baselines3/common/off_policy_algorithm.py:367-399; 032_00:244-258", "finding": "SB3 warmup uses action_space.sample(); epsilon exploration also samples without env.action_masks().", "scope": "current 032 path", "required_adapter": "Replace standard DQN collection with select_action(mask) or a proven mask-aware implementation."},
        {"candidate": "032_00_standard_sb3_dqn", "phase": "greedy action during training", "mask_safe": "NO", "buffer_action_matches_executed_action": "NO_OR_UNSAFE", "evidence": "stable_baselines3/dqn/dqn.py:233-261; 032_00:244-258", "finding": "Policy.predict has no dynamic mask argument; masked_greedy_action is called only by evaluation code at 032_00:278-288,333-370.", "scope": "current 032 path", "required_adapter": "Apply the current-state mask inside training-time action selection."},
        {"candidate": "032_00_standard_sb3_dqn", "phase": "Bellman target", "mask_safe": "NO", "buffer_action_matches_executed_action": "NO_OR_UNSAFE", "evidence": "stable_baselines3/dqn/dqn.py:187-208", "finding": "next_q_values.max(dim=1) is taken over all actions; no next-state mask is sampled or applied.", "scope": "current 032 path", "required_adapter": "Store next_masks and mask target maximization before max()."},
        {"candidate": "032_00_standard_sb3_dqn", "phase": "replay/action-effect closure", "mask_safe": "NO", "buffer_action_matches_executed_action": "NO_OR_UNSAFE", "evidence": "stable_baselines3/common/off_policy_algorithm.py:449-507,514-591; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:130-145", "finding": "The wrapper can map an invalid requested index to no-op before DSSAT; SB3 stores buffer_actions, not a verified executed/mapped action. Existing 032 eval had 95 rows and zero observed fallback, but that does not repair the unsafe training path.", "scope": "current 032 path", "required_adapter": "Reject invalid actions before env.step, or store the actual mapped action and assert action/reward/next_state closure."},
        {"candidate": "031_19_standard_sb3_dqn", "phase": "all three training paths", "mask_safe": "NO", "buffer_action_matches_executed_action": "NO_OR_UNSAFE", "evidence": "src/run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19.py:210-230,298-330; same SB3 DQN runtime", "finding": "It is the same standard SB3 DQN pattern; masked_greedy_action is evaluation-only and reward contract also differs from current 055.", "scope": "historical free-timing path", "required_adapter": "Do not use 031_19 as current YC matched DQN."},
        {"candidate": "029_mask_aware_dqn", "phase": "exploration", "mask_safe": "YES_IN_LEGACY_SCOPE", "buffer_action_matches_executed_action": "YES_IN_LEGACY_SCOPE", "evidence": "src/mask_aware_dqn_029.py:150-157; benchmark_results/029_01_sy2014_mask_aware_dqn_smoke/029_01_result.json", "finding": "Random epsilon choices are sampled only from valid indices; smoke reports all collected actions valid and zero invalid attempts.", "scope": "legacy six-stage/9-action env", "required_adapter": "Reuse the core only after setting action_dim=16 and binding current YC masks."},
        {"candidate": "029_mask_aware_dqn", "phase": "greedy action", "mask_safe": "YES_IN_LEGACY_SCOPE", "buffer_action_matches_executed_action": "YES_IN_LEGACY_SCOPE", "evidence": "src/mask_aware_dqn_029.py:150-157; src/run_five_site_stage_mask_aware_dqn_029_02.py:105-118", "finding": "Invalid Q entries are set to -inf before argmax; runner passes the current env mask.", "scope": "legacy six-stage/9-action env", "required_adapter": "Keep one mask source for train and deterministic evaluation."},
        {"candidate": "029_mask_aware_dqn", "phase": "Bellman target", "mask_safe": "YES_IN_LEGACY_SCOPE", "buffer_action_matches_executed_action": "YES_IN_LEGACY_SCOPE", "evidence": "src/mask_aware_dqn_029.py:162-183", "finding": "next_masks are stored and applied before target max; terminal transitions disable bootstrap.", "scope": "legacy six-stage/9-action env", "required_adapter": "Add current YC next_mask shape/assertions and preserve terminal handling."},
        {"candidate": "029_mask_aware_dqn", "phase": "replay/action-effect closure", "mask_safe": "YES_IN_LEGACY_SCOPE", "buffer_action_matches_executed_action": "YES_IN_LEGACY_SCOPE", "evidence": "src/mask_aware_dqn_029.py:47-83; src/run_five_site_stage_mask_aware_dqn_029_02.py:105-143; benchmark_results/029_01_sy2014_mask_aware_dqn_smoke/029_01_result.json", "finding": "Replay rejects invalid collection actions and non-terminal empty next masks; runner stores the selected action and computes next_mask from the stepped env. Engineering smoke passed 60/60 transition and closure checks.", "scope": "legacy six-stage/9-action env", "required_adapter": "Instrument requested/mapped/executed action and reward closure in the daily YC adapter."},
    ]


def hyperparameter_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    standard = {
        "candidate": "032_00_standard_sb3_dqn",
        "scope": "LCA2010 smoke; not a YC-ready DQN",
        "source": "experiments/ppo_observed_years/config_032_00_free_timing_stress_aware_ppo_dqn_smoke.yaml; src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:211-216,244-258",
        "effective_status": "passed_to_sb3" ,
    }
    for parameter, value in [
        ("total_timesteps", "5000"), ("learning_rate", "0.00001"), ("buffer_size", "100000"),
        ("learning_starts", "1024"), ("batch_size", "1024"), ("gamma", "0.99"),
        ("train_freq", "4"), ("gradient_steps", "1"), ("target_update_interval", "120"),
        ("exploration_fraction", "1.0"), ("exploration_initial_eps", "1.0"),
        ("exploration_final_eps", "0.0"), ("max_grad_norm", "10.0"),
        ("net_arch", "[256,256,256]"), ("weight_decay", "0.001"),
    ]:
        rows.append({**standard, "parameter": parameter, "value": value, "freeze_candidate": "NO", "notes": "These values are historical 032 smoke settings and must not be silently transplanted to current YC."})
    custom = {
        "candidate": "029_mask_aware_dqn",
        "scope": "legacy stage env; reusable core candidate",
        "source": "src/mask_aware_dqn_029.py:100-134; src/run_five_site_stage_mask_aware_dqn_029_02.py:105-143",
        "effective_status": "effective_in_custom_runner",
    }
    for parameter, value, notes in [
        ("learning_rate", "0.0003", "Adam"), ("gamma", "1.0", "matches current PPO gamma but must be explicitly registered"),
        ("replay_capacity", "10000", "replay stores masks and next_masks"), ("batch_size", "30", "custom train_step"),
        ("network", "obs_dim -> 32 -> 32 -> action_dim", "vanilla MLP; not SB3 policy"),
        ("epsilon", "1.0 to 0.05 over 70% of 240 steps", "valid actions only"),
        ("gradient_clip", "10.0", "torch.nn.utils.clip_grad_norm_"),
        ("target_sync", "every 60 environment steps in 029_02", "four syncs over 240 steps"),
        ("checkpoint_schedule", "0,60,120,180,240", "custom .pt checkpoints"),
        ("action_dim", "9 default", "must become 16 for YC current contract"),
    ]:
        rows.append({**custom, "parameter": parameter, "value": value, "freeze_candidate": "CONDITIONAL", "notes": notes})
    rows.append({"candidate": "029_mask_aware_dqn", "scope": "legacy stage env; reusable core candidate", "source": "src/mask_aware_dqn_029.py:197-228", "effective_status": "limitation", "parameter": "replay_buffer_checkpointing", "value": "not saved by save()", "freeze_candidate": "CONDITIONAL", "notes": "Evaluation checkpoint round-trip passed; resume-from-replay semantics are not available without an adapter."})
    return rows


def historical_rows() -> list[dict[str, object]]:
    return [
        {"run": "032_00", "implementation": "standard SB3 DQN", "site_year_scope": "LCA2010", "seeds": "0", "timesteps_or_steps": "5000 env timesteps", "checkpoint_evidence": "final .zip", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "single site-year; standard DQN is mask-unsafe during training; reward/budget not current 055", "evidence": "benchmark_results/032_00_free_timing_stress_aware_ppo_dqn_smoke"},
        {"run": "031_19", "implementation": "standard SB3 DQN", "site_year_scope": "SYA2014", "seeds": "0", "timesteps_or_steps": "20000 env timesteps", "checkpoint_evidence": "final .zip", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "same unsafe SB3 path; different reward and site-year", "evidence": "benchmark_results/031_19_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_seed0"},
        {"run": "031_13", "implementation": "standard SB3 DQN", "site_year_scope": "SYA2014", "seeds": "0,1,2", "timesteps_or_steps": "literature DQN interval7 smoke/formal record", "checkpoint_evidence": "final .zip per seed", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "same standard DQN mask gap; no current YC contract", "evidence": "benchmark_results/031_13_literature_dqn_interval7_ncost2x"},
        {"run": "031_15", "implementation": "standard SB3 DQN", "site_year_scope": "SYA2014", "seeds": "0", "timesteps_or_steps": "100000 env timesteps", "checkpoint_evidence": "final .zip", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "longer budget does not repair the algorithmic mask path", "evidence": "benchmark_results/031_15_literature_dqn_interval7_ncost2x_100k_seed0"},
        {"run": "029_01", "implementation": "custom mask-aware DQN", "site_year_scope": "SY2014 six-stage", "seeds": "0", "timesteps_or_steps": "60 env steps; 10 seasons", "checkpoint_evidence": "checkpoint_000000.pt and checkpoint_000060.pt", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "excellent legacy engineering closure, but action_dim=9 and stage timing", "evidence": "benchmark_results/029_01_sy2014_mask_aware_dqn_smoke/029_01_result.json"},
        {"run": "029_02", "implementation": "custom mask-aware DQN", "site_year_scope": "five-site six-stage", "seeds": "0,1,2", "timesteps_or_steps": "240 env steps per site-seed", "checkpoint_evidence": "0,60,120,180,240 .pt checkpoints", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "YC exists historically, but it is not the current daily 055 YC Original contract", "evidence": "benchmark_results/029_02_five_site_stage_mask_aware_dqn/YC"},
        {"run": "029_03", "implementation": "custom mask-aware DQN frozen evaluation", "site_year_scope": "YC2008 among 12 cross-year cases", "seeds": "0,1,2", "timesteps_or_steps": "frozen eval; no new training", "checkpoint_evidence": "stage_actions.csv and result.json", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "cross-year stage evidence, not current YC daily 10-year split", "evidence": "benchmark_results/029_03_frozen_maskaware_dqn_crossyear/cases/YC2008"},
        {"run": "029_04", "implementation": "custom mask-aware DQN replay/evidence", "site_year_scope": "FQ2016 and related legacy cases", "seeds": "varies", "timesteps_or_steps": "frozen daily replay/evidence", "checkpoint_evidence": "result.json and stage_actions.csv", "historical_evidence_available": "YES", "sufficient_for_current_yc_checkpoint_reaudit": "NO", "reason": "legacy environment and metric/provenance scope", "evidence": "benchmark_results/029_04_dqn_daily_replay"},
    ]


def grid_rows() -> list[dict[str, object]]:
    irrigation = [0, 15, 30, 45]
    nitrogen = [0, 40, 80, 120]
    return [{"action_index": i_index * len(nitrogen) + n_index, "irrigation_mm": i, "nitrogen_kg_ha": n, "mapping": "row-major i_index*4+n_index", "legal_at_runtime": "mask-dependent"} for i_index, i in enumerate(irrigation) for n_index, n in enumerate(nitrogen)]


def dependency_rows() -> list[dict[str, object]]:
    return [
        {"layer": "current PPO reference", "module": "src/run_055_yca_lowIC_site_transfer/run_055_00_yca_lowIC_expanded_action_maskableppo.py", "role": "YC lowIC preflight, 10+10 years, 16 actions, current formal runner", "status": "reference"},
        {"layer": "current PPO engine", "module": "src/run_sya_lowIC_binary_timing_maskableppo_042_10.py", "role": "patches current wrapper and launches batch training", "status": "reference"},
        {"layer": "current safety/reward", "module": "src/run_sya_lowIC_ppo_late_irrigation_reserve_mask_040_36.py -> 040_28 -> 040_26", "role": "late reserve, SWFAC guardrail, I240 staged mask", "status": "reference"},
        {"layer": "unsafe DQN runner", "module": "src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py", "role": "standard SB3 DQN plus evaluation-only manual mask", "status": "not ready"},
        {"layer": "unsafe DQN runner", "module": "src/run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19.py", "role": "standard SB3 free-timing DQN", "status": "not ready"},
        {"layer": "reusable DQN core", "module": "src/mask_aware_dqn_029.py", "role": "mask-safe epsilon, greedy, Bellman target, replay masks", "status": "minimal adapter candidate"},
        {"layer": "legacy custom runner", "module": "src/run_five_site_stage_mask_aware_dqn_029_02.py", "role": "six-stage, 9-action training/checkpoint/eval loop", "status": "not current contract"},
    ]


def build_report() -> str:
    return """# YC DQN readiness audit v2

审计日期：2026-09-11  
审计范围：只读审计；未训练、未修改现有正式 PPO/DQN 代码、未覆盖既有结果。独立产物目录为 `experiments/225_yc_dqn_readiness_audit/`。

## 结论先行

最终分类：`READY_WITH_MINIMAL_ADAPTER`

最终建议：`Proceed to matched PPO–DQN comparison: YES`，但这个 YES 仅表示进入“最小适配器实现 → smoke gate → 再决定正式训练”的流程；不表示可以直接运行当前 032/031 标准 SB3 DQN，也不表示本轮已训练 DQN。

核心原因：

1. `032_00` 和 `031_19` 使用标准 SB3 `DQN`。它们的训练探索、训练时 greedy、Bellman target 都没有使用动态 action mask；032 只有 evaluation 的手写 `masked_greedy_action`。
2. `src/mask_aware_dqn_029.py` 已经实现了所需的三处 mask 和 replay mask，并有 `029_01` 的工程 smoke 通过证据；但它默认是旧的六阶段、9-action、stage observation 契约，不能直接当作当前 YC Original daily 16-action DQN。
3. 因此最小适配是“复用 mask-aware 核心 + 接入当前 055 YC lowIC daily env/reward/safety/mapping”，不是重做算法，也不是把 032 DQN 的结果拿来比较。

## 1. 仓库中发现的 DQN 路径

### 标准 SB3 路径：不满足当前 mask 安全要求

- `src/run_free_timing_stress_aware_ppo_dqn_smoke_032_00.py:244-258`：实例化 `stable_baselines3.DQN`；`action_masks()` 只在 wrapper 和 evaluation 手写 greedy 中出现。
- `src/run_free_timing_discrete_masked_dqn_ncost2x_scaled_reward_031_19.py:210-230`：同类标准 SB3 DQN；evaluation 也单独调用手写 masked greedy。
- 本地 SB3 2.7.1 的关键行为：`common/off_policy_algorithm.py:367-399` 在 warmup 使用 `action_space.sample()`；`dqn/dqn.py:233-261` 的 predict 不接动态 mask；`dqn/dqn.py:187-208` 对 target Q 在所有动作上直接 `max`。

### 自定义 mask-aware 路径：安全核心可复用，但契约不匹配

- `src/mask_aware_dqn_029.py:150-157`：epsilon 随机和 greedy 只从/选择合法动作。
- `src/mask_aware_dqn_029.py:162-183`：Bellman target 在 `next_masks` 上先屏蔽再 max，terminal 不 bootstrap。
- `src/mask_aware_dqn_029.py:47-83`：replay 同时保存当前/下一状态 mask，并拒绝非法 collection action 与非 terminal 空 next mask。
- `src/run_five_site_stage_mask_aware_dqn_029_02.py:105-143`：旧版 runner 将当前 mask、action、step 后的 next mask 放入 transition。
- `benchmark_results/029_01_sy2014_mask_aware_dqn_smoke/029_01_result.json`：60 steps、60 个合法 action、zero invalid attempts、一次 optimizer update、一次 target sync、checkpoint round-trip 等工程检查均通过。

这组通过证据证明的是旧 stage/9-action 路径的工程闭环，不是当前 YC 055 daily/16-action readiness。

## 2. 当前 YC Original PPO reference contract

当前对照基准是 `configs/055_00_yca_lowIC_expanded_action_maskableppo.json` 及其 manifest，而不是名称相似的历史目录：

| 契约项 | 当前 reference |
|---|---|
| 站点/输入 | YCA/YC，`lowIC`，YC `CNYC0801.MZX`，`MZCER048.CUL` |
| 年份 | train 2004–2013；validation 2014–2023 |
| 观测 | `046_02_raw_observation`，raw/no normalization，no weather forecast，daily state |
| 动作 | 16 个 joint actions：I `[0,15,30,45]` × N `[0,40,80,120]`，row-major |
| 日约束 | I≤45 mm/day；N≤120 kg/ha/day；I/N 最小间隔均 7 天 |
| DAP 窗口 | irrigation `[1,120]`；fertilization `[1,90]` |
| 季节/阶段安全 | I=240 season cap；DAP≤30/60/90 阶段上限 75/150/195；late reserve 45 mm |
| reward/safety | 当前 040_28 SWFAC threshold 0.05、penalty coefficient 50，并沿 040_26/040_36 reserve chain |
| 输出 | 每个站点/年份/seed/checkpoint 的 daily/output/provenance；yield、I、N、PFP_N；WP_ET/NUE 需一致的 Summary.OUT/ETCP replay |
| checkpoint | 25K/50K/75K/100K；正式 report checkpoint 100K |

032 的静态 action grid 与日窗口/最小间隔相似，但 I seasonal budget、staged reserve、SWFAC reward、YC 年份/输入和评估范围均不相同。

## 3. 四项 mask readiness gate

| gate | 032/031 标准 SB3 DQN | 029 自定义核心（旧契约） | 当前 YC 结论 |
|---|---|---|---|
| exploration 只探索合法 action | `NO` | `YES_IN_LEGACY_SCOPE` | 未通过直接复用；适配器必须接当前 YC mask |
| greedy argmax 只选合法 action | `NO`（训练时）；eval 手写路径才 mask | `YES_IN_LEGACY_SCOPE` | 未通过直接复用 |
| Bellman target 只在 next-state legal action 上 max | `NO` | `YES_IN_LEGACY_SCOPE` | 未通过直接复用 |
| replay action == 实际执行 action/reward/next_state | `NO_OR_UNSAFE`；SB3 存 buffer action，wrapper 可把非法请求 fallback 为 no-op | `YES_IN_LEGACY_SCOPE` | 必须在 YC daily adapter 中重新证明 |

032 的一份已有 evaluation daily 文件有 95 行，观测到 `mask_forced_noop=0`、mapped action difference=0、safety trigger=0；这只能说明该次 deterministic eval 没触发 fallback，不能证明标准 SB3 训练 collection 与 Bellman update 是 mask-safe。

## 4. action grid 与 mapping

当前 YC 16-action mapping 已固化到 `results/yc_dqn_action_grid_mapping.csv`：

`index = irrigation_index * 4 + nitrogen_index`

因此 index 0–3 是 I=0 的四个 N 档，index 4–7 是 I=15，index 8–11 是 I=30，index 12–15 是 I=45。mask 是 state-dependent；index 存在不等于当时合法。

029 的默认 `action_dim=9`，其 stage runner 也明确构造 `next_mask=np.zeros(9, dtype=bool)`，所以不能加载/解释为当前 16-action YC DQN。

## 5. DQN hyperparameter inventory

完整清单在 `results/yc_dqn_hyperparameter_inventory.csv`。

- 032 historical SB3 settings：learning rate `1e-5`、buffer `100000`、learning starts `1024`、batch `1024`、gamma `0.99`、train frequency `4`、gradient steps `1`、target update `120`、epsilon `1.0→0.0` over fraction `1.0`、MLP `[256,256,256]`、weight decay `0.001`。
- 029 custom core settings：Adam learning rate `3e-4`、gamma `1.0`、replay `10000`、batch `30`、MLP `32→32`、epsilon `1.0→0.05` over 70% of 240 steps、gradient clip `10`、target sync every 60 env steps、checkpoint `0/60/120/180/240`。
- 本审计不把任何一套旧超参直接宣布为 YC formal DQN 超参。适配器 smoke 前只冻结“契约和 mask 行为”，正式 DQN 的 seed/checkpoint/budget 需要在 smoke PASS 后另行登记。
- 029 `save()` 不保存 replay buffer；checkpoint evaluation round-trip 已通过，但若要求 resume-from-replay，需要单独适配。

## 6. 历史证据与 checkpoint re-audit 判断

历史 DQN 产物确实存在，完整目录索引在 `results/yc_historical_dqn_inventory.csv`。其中 `029_01/029_02/029_03` 对自定义旧版 DQN 的 mask/transition/evaluation 有可复核证据；`032/031` 对标准 SB3 DQN 有模型、daily 和 eval 输出。

但没有一份历史产物同时满足“当前 YC 055 的 lowIC、2004–2013 train、2014–2023 validation、daily raw observation、16-action、I240 staged reserve、040_28/040_36 reward/safety、当前 checkpoint/output contract”。所以历史产物可用于实现审计和回归参考，不能替代当前 YC checkpoint re-audit，也不能支持直接进入正式 matched DQN training。

## 7. 最小适配器与下一步（本轮不执行训练）

建议只做以下最小改动后重新 smoke：

1. 复用 `MaskAwareDQN` 核心，显式设置当前 YC observation dimension 与 `action_dim=16`。
2. 接入 055_00 的 YC lowIC daily env、当前 reward/safety wrapper chain、10+10 年份选择和相同 cultivar/input provenance。
3. 使用同一个 row-major action mapping 生成 `action_masks()`、`select_action()`、replay action、evaluation action 和 daily log。
4. transition 写入前断言：selected action 合法；env 实际执行 index 等于 replay action；reward/next observation/next mask 来自同一次 `env.step`；terminal next mask 不用于 bootstrap。
5. 先做小 smoke gate：至少覆盖 invalid-mask rejection、epsilon legality、greedy legality、masked Bellman target、transition closure、16-action mapping、YC wrapper 约束和 checkpoint reload；任一失败即停止，不启动长训练。
6. smoke PASS 后再制定与 PPO 相同站点/年份/seed/checkpoint/output 口径的 matched comparison；保留所有失败记录。

## 8. 对 prompt 要求的逐项回答

1. **存在哪些 DQN 实现？** 标准 SB3 DQN（032/031 等）和项目自定义 `MaskAwareDQN`（029 系列）；另有历史 replay/evaluation runner。
2. **哪些路径可复用？** 029 的 mask-aware 核心可复用；032/031 的标准 DQN 训练路径不可复用为当前 matched DQN。
3. **当前 YC PPO contract 是什么？** 见本报告第 2 节及 contract CSV：YCA/YC lowIC，2004–2013/2014–2023，raw daily observation，16 actions，I240 staged reserve，N250，7-day intervals，040_28/040_36 reward/safety chain。
4. **探索是否只选合法动作？** 032/031：否；029 旧契约：是。
5. **greedy 是否只选合法动作？** 032/031 训练时否，evaluation 手写路径是；029 旧契约是。
6. **Bellman target 是否 mask？** 032/031 否；029 使用 `next_masks`，是（旧契约）。
7. **replay action 是否等于实际执行 action？** 032/031 不能保证，标准 SB3 存 buffer action 而 wrapper 可 fallback；029 旧 runner 的 60-step smoke 通过。
8. **action grid/mapping 是否一致？** 032 静态 16-grid 的排列与当前 YC 一致；029 默认 9-action 不一致；当前 DQN 尚未绑定 YC wrapper。
9. **主要 contract mismatch 是什么？** YC 年份/输入、daily vs stage、16 vs 9、I240/staged reserve、040_28/040_36 reward/safety、checkpoint/output protocol。
10. **超参是什么？** 见 hyperparameter CSV；032 与 029 分列，未将旧超参直接宣布为 YC formal 配置。
11. **历史 evidence/checkpoint re-audit 是否足够？** 历史证据足够审计旧实现，但不足以完成当前 YC checkpoint re-audit。
12. **readiness classification？** `READY_WITH_MINIMAL_ADAPTER`。
13. **Proceed to matched PPO–DQN comparison？** `YES`，条件是最小适配器 smoke PASS；当前 032/031 标准 SB3 DQN 不能直接训练/比较，本轮未训练。

## Evidence index

- `results/yc_dqn_readiness_contract_audit.csv`
- `results/yc_dqn_action_mask_audit.csv`
- `results/yc_dqn_hyperparameter_inventory.csv`
- `results/yc_historical_dqn_inventory.csv`
- `results/yc_dqn_action_grid_mapping.csv`
- `results/yc_dqn_code_dependency_map.csv`
- `scripts/build_audit.py`
"""


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    write_csv("yc_dqn_readiness_contract_audit.csv", contract_rows())
    write_csv("yc_dqn_action_mask_audit.csv", mask_rows())
    write_csv("yc_dqn_hyperparameter_inventory.csv", hyperparameter_rows())
    write_csv("yc_historical_dqn_inventory.csv", historical_rows())
    write_csv("yc_dqn_action_grid_mapping.csv", grid_rows())
    write_csv("yc_dqn_code_dependency_map.csv", dependency_rows())
    report_path = DOCS / "yc_dqn_readiness_audit.md"
    report_path.write_text(build_report(), encoding="utf-8")
    manifest = {
        "audit": "yc_dqn_readiness_audit_v2",
        "date": "2026-09-11",
        "classification": "READY_WITH_MINIMAL_ADAPTER",
        "proceed_to_matched_ppo_dqn_comparison": "YES_AFTER_ADAPTER_SMOKE",
        "training_performed": False,
        "formal_code_modified": False,
        "outputs": [str(path.relative_to(ROOT)) for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "audit_manifest.json"],
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
