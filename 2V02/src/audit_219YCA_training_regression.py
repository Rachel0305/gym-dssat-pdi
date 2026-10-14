"""219YCA: reproducible, CPU-only diagnosis of the saved 218YCA policies."""
from __future__ import annotations

import argparse
import atexit
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import weakref

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import psutil
import torch
from sb3_contrib import MaskablePPO
import audit_218YCA_action_counterfactual_5k as ref

torch.set_num_threads(1)
CHECKPOINTS = [5000, 100000, 200000]
DEFAULT_OUT = ROOT / "benchmark_results/219YCA_training_regression_diagnosis"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def rss():
    return psutil.Process().memory_info().rss / 1048576


def release_env(env):
    """Only unregister this closed environment's own exit callback."""
    raw = env.unwrapped
    env.close()
    callback = getattr(raw, "_cleanup_process", None)
    if callback is not None:
        atexit.unregister(callback)


def setup(out):
    contract, config, env_config = ref.build_env()
    env_config = copy.deepcopy(env_config)
    env_config.setdefault("paths", {})["output_root"] = out.relative_to(ROOT).as_posix()
    ref.runner.base03222.base.StressAwareDiscreteWrapper = ref.ProjectProfitDiscreteWrapper
    ref.ppo_safe_rendering.MULTISITE_INPUT_ROOT = ROOT / contract["input_root"]
    return contract, config, env_config


def make_env(config, env_config, year, tag, seed=0, evaluation=True):
    return ref.runner.base03222.base.make_env(config, env_config, "YCA", int(year), seed, tag, evaluation=evaluation)


@torch.no_grad()
def forward(model, observations, masks):
    tensor, _ = model.policy.obs_to_tensor(np.asarray(observations, dtype=np.float32))
    dist = model.policy.get_distribution(tensor, action_masks=np.asarray(masks, dtype=bool))
    probs = dist.distribution.probs.cpu().numpy()
    values = model.policy.predict_values(tensor).cpu().numpy().reshape(-1)
    features = model.policy.extract_features(tensor)
    net = model.policy.mlp_extractor.policy_net
    first = net[1](net[0](features))
    second = net[3](net[2](first))
    return probs, values, (first.abs() > .99).float().mean(dim=1).cpu().numpy(), (second.abs() > .99).float().mean(dim=1).cpu().numpy()


def feature_names(env, obs):
    # At reset the legacy scalar fallback zips 17 variable names with 22
    # flattened values, before history is populated. Use the raw dictionary.
    state = env.unwrapped.observation
    names = []
    for key in env.unwrapped.observation_variables:
        arr = np.asarray(state[key]).reshape(-1)
        names.extend([key] if len(arr) == 1 else [f"{key}_{i}" for i in range(len(arr))])
    assert len(names) == np.asarray(obs).size, (names, np.asarray(obs).shape)
    values = np.asarray(env.unwrapped.observation_dict_to_array(state), dtype=np.float32)
    np.testing.assert_allclose(values, obs, rtol=0, atol=0)
    return names


def replay(model, checkpoint, year, out, config, env_config, expected=None):
    env = make_env(config, env_config, year, f"219YCA_ck{checkpoint}_{year}")
    rows, observations, masks = [], [], []
    try:
        obs, _ = env.reset()
        names = feature_names(env, obs)
        grid = np.asarray([[g["amir"], g["anfer"]] for g in env.grid])
        assert grid.shape == (16, 2)
        done = False
        for t in range(260):
            before = dict(env.unwrapped.observation)
            dap = env._dap()
            mask = env.action_masks()
            x = np.asarray(obs, dtype=np.float32).reshape(-1)
            assert np.isfinite(x).all()
            p, v, sat1, sat2 = forward(model, x[None], mask[None])
            p = p[0]
            action = int(p.argmax())
            assert mask[action] and abs(p.sum() - 1) < 1e-5
            legal = int(mask.sum())
            entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            state = env.safety_state
            last_i = state.last_irrigation_dap
            ir_legal = bool(mask[grid[:, 0] > 0].any())
            row = dict(checkpoint=checkpoint, year=year, step_index=t, dap=dap,
                       legal_count=legal, mask="".join(map(str, mask.astype(int))), action=action,
                       top_probability=float(p.max()), top_gap=float(np.sort(p)[-1]-np.sort(p)[-2]),
                       entropy_normalized=entropy/np.log(legal) if legal > 1 else np.nan,
                       value=float(v[0]), first_tanh_saturated=float(sat1[0]), second_tanh_saturated=float(sat2[0]),
                       irrigation_legal=ir_legal, irrigation_gap=dap-last_i if last_i is not None else np.nan,
                       irrigation_unlock=bool(ir_legal and last_i is not None and dap-last_i == config["action_safety"]["min_days_between_irrigation"]),
                       previous_i=float(state.cumulative_irrigation), previous_n=float(state.cumulative_n))
            row.update({f"obs_{name}": float(value) for name, value in zip(names, x)})
            row.update({f"p_{i}": float(q) for i, q in enumerate(p)})
            for key in ("swfac", "nstres", "grnwt", "topwt", "xlai"):
                row[f"pre_{key}"] = ref.scalar(before.get(key))
            observations.append(x.copy())
            masks.append(mask.copy())
            obs, reward, term, trunc, _ = env.step(action)
            info = dict(env.last_action_info)
            row.update(reward=float(reward), irrigation=float(info["safe_action_amir"]), nitrogen=float(info["safe_action_anfer"]),
                       forced_noop=bool(info["mask_forced_noop"]), safety_trigger=str(info["action_safety_triggered"]),
                       requested_i=float(grid[action, 0]), requested_n=float(grid[action, 1]))
            rows.append(row)
            done = bool(term or trunc)
            if done:
                break
        if not done:
            raise RuntimeError("Episode failed to finish")
        data = pd.DataFrame(rows)
        data["return_G"] = data.reward.iloc[::-1].cumsum().iloc[::-1]
        data["G_minus_V"] = data.return_G - data.value
        final_yield = ref.scalar(env.last_obs_dict.get("grnwt"))
        I, N = float(data.irrigation.sum()), float(data.nitrogen.sum())
        profit = final_yield - 1.1 * I - 1.58 * N
        assert abs(data.reward.sum() - profit * .001) < 1e-4
        assert not data.forced_noop.any() and (data.safety_trigger == "").all()
        np.testing.assert_allclose(data[["irrigation", "nitrogen"]], data[["requested_i", "requested_n"]], atol=1e-8)
        signature = ";".join(f"{r.dap}:I{r.irrigation:g}/N{r.nitrogen:g}" for r in data.itertuples() if r.irrigation+r.nitrogen > 0)
        mismatch = 0
        if expected is not None:
            assert len(expected) == len(data), (len(expected), len(data))
            mismatch = int((expected.discrete_action_index.to_numpy() != data.action.to_numpy()).sum())
            assert mismatch == 0, f"Saved action mismatch {checkpoint}, {year}: {mismatch}"
            assert abs(final_yield - float(expected.grnwt.iloc[-1])) <= 1.1
        snapshot = out / "snapshots" / str(checkpoint) / str(year)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ref.snapshot_from_env(env), snapshot)
        metrics = ref.FIG055.baseline.metrics_from_snapshot(snapshot, final_yield)
        result = dict(checkpoint=checkpoint, year=year, yield_kg_ha=final_yield, irrigation=I, nitrogen=N,
                      simple_profit_score=profit, WP_ET=metrics["WP_ET_kg_m3"], PFP_N=final_yield/N if N>0 else np.nan,
                      irrigation_events=int((data.irrigation>0).sum()), n_events=int((data.nitrogen>0).sum()),
                      action_signature=signature, reward_identity_error=float(data.reward.sum()-profit*.001),
                      saved_action_mismatch=mismatch, steps=len(data), rss_mb=rss())
        data.to_csv(out / "tables" / f"decisions_ck{checkpoint}_{year}.csv", index=False)
        print(f"[219YCA replay] ck={checkpoint} year={year} yield={final_yield:.1f} I={I:g} N={N:g} rss={rss():.1f}MB", flush=True)
        return data, np.stack(observations), np.stack(masks), result, names
    finally:
        release_env(env)


def memory_probe(out, config, env_config):
    refs = []
    for j in range(4):
        env = make_env(config, env_config, 2014, f"219YCA_memory_probe{j}")
        raw = env.unwrapped
        refs.append(weakref.ref(raw))
        env.close()
        del raw, env
        gc.collect()
    retained_before = sum(r() is not None for r in refs)
    callback_names = []
    for w in refs:
        obj = w()
        if obj is not None:
            callback = getattr(obj, "_cleanup_process", None)
            if callback is not None:
                callback_names.append(callback.__qualname__)
                atexit.unregister(callback)
            del callback
        del obj
    gc.collect()
    retained_after = sum(r() is not None for r in refs)
    result = dict(created=4, retained_after_normal_close_gc=retained_before,
                  retained_after_own_atexit_unregister_gc=retained_after, callbacks=callback_names,
                  interpretation="Controlled reachability test; does not by itself assign all RSS growth to these objects.")
    write_json(out / "memory_lifecycle_probe.json", result)
    return result


def offline_checks(out, data, X, M, names, models):
    common = data.checkpoint.eq(5000).to_numpy()
    common &= data.legal_count.ge(2).to_numpy()
    common_rows = data.loc[common].copy().reset_index(drop=True)
    probs_rows, sensitivity = [], []
    water_n_idx = [j for j,n in enumerate(names) if n.startswith(("sw_", "swfac", "nstres", "no3", "nh4", "sno3", "snh4"))]
    for checkpoint, model in models.items():
        p, v, s1, s2 = forward(model, X[common], M[common])
        for i, row in common_rows.iterrows():
            probs_rows.append(dict(model_checkpoint=checkpoint, year=int(row.year), dap=int(row.dap), step_index=int(row.step_index),
                                   selected_action=int(p[i].argmax()), value=float(v[i]), saturation=float(s1[i]),
                                   **{f"p_{a}": float(q) for a,q in enumerate(p[i])}))
        # Same DAP; donor is another year's observed vector. Mask stays fixed.
        donor = np.arange(len(common_rows))
        for _, indexes in common_rows.groupby("dap").groups.items():
            idx = np.asarray(list(indexes))
            for k in idx:
                candidates = idx[common_rows.loc[idx,"year"].to_numpy() != common_rows.loc[k,"year"]]
                if len(candidates):
                    donor[k] = candidates[0]
        eligible = donor != np.arange(len(donor))
        for mode in ("all_observed_state_same_dap", "water_n_features_only"):
            altered = X[common].copy()
            if mode == "all_observed_state_same_dap":
                altered = altered[donor]
            elif water_n_idx:
                altered[:,water_n_idx] = altered[donor][:,water_n_idx]
            else:
                continue
            q, _, _, _ = forward(model, altered, M[common])
            changed_input = np.any(altered != X[common],axis=1) & eligible
            changed_action = p.argmax(axis=1) != q.argmax(axis=1)
            l1 = np.abs(p-q).sum(axis=1)
            sensitivity.append(dict(checkpoint=checkpoint, intervention=mode, eligible_rows=int(changed_input.sum()),
                                    changed_action_rows=int(changed_action[changed_input].sum()),
                                    action_change_rate=float(changed_action[changed_input].mean()) if changed_input.any() else np.nan,
                                    mean_probability_l1=float(l1[changed_input].mean()) if changed_input.any() else np.nan,
                                    max_probability_l1=float(l1[changed_input].max()) if changed_input.any() else np.nan))
    pd.DataFrame(probs_rows).to_csv(out/"tables/common_5k_states_all_models.csv",index=False)
    pd.DataFrame(sensitivity).to_csv(out/"tables/fixed_mask_state_sensitivity.csv",index=False)
    profile=[]
    for j,name in enumerate(names):
        values=X[:,j]
        profile.append(dict(index=j,feature=name,min=float(values.min()),median=float(np.median(values)),p95_abs=float(np.quantile(abs(values),.95)),max=float(values.max()),std=float(values.std())))
    pd.DataFrame(profile).to_csv(out/"tables/observation_profile.csv",index=False)
    return sensitivity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--out",default=str(DEFAULT_OUT))
    args=parser.parse_args()
    out=Path(args.out).resolve()
    if not out.is_relative_to(ROOT) or out.exists():
        raise ValueError("New project-local output directory required")
    (out/"tables").mkdir(parents=True)
    contract,config,env_config=setup(out)
    checkpoints=[5000] if args.smoke else CHECKPOINTS
    years=[2014] if args.smoke else list(range(2014,2024))
    models={}
    provenance={}
    for step in checkpoints:
        path=ref.PPO_ROOT/"models/YCA"/f"YCA_half_split_stress_aware_maskableppo_seed0_ckpt{step}.zip"
        model=MaskablePPO.load(str(path),device="cpu")
        model.policy.set_training_mode(False)
        models[step]=model
        provenance[str(step)]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),n_steps=model.n_steps,batch_size=model.batch_size,learning_rate=model.learning_rate,policy=str(model.policy))
    write_json(out/"contract.json",dict(source_contract=contract,engine_config=config,models=provenance,
               script_sha256=sha(__file__),effective_wrapper="ProjectProfitDiscreteWrapper",stress_reward_used=False,
               obsolete_summary_fields=["swfac_guardrail_process_penalty", "late_irrigation_reserve_mask (not enforced by current profit wrapper)"],
               input_root=str(ref.ppo_safe_rendering.MULTISITE_INPUT_ROOT),validation_role="development; not independent final test"))
    frames,obs_list,mask_list,episodes=[],[],[],[]
    for ck,model in models.items():
        for year in years:
            if rss()>1536:
                raise MemoryError("Diagnostic RSS warning exceeded; refuse additional episodes")
            expected=pd.read_csv(ref.MODEL_DAILY_ROOT/f"YCA_{year}_seed0_ckpt{ck}_daily.csv",keep_default_na=False)
            data,x,m,result,names=replay(model,ck,year,out,config,env_config,expected)
            frames.append(data);obs_list.append(x);mask_list.append(m);episodes.append(result)
    data=pd.concat(frames,ignore_index=True)
    X=np.concatenate(obs_list);M=np.concatenate(mask_list)
    data.to_csv(out/"tables/decision_states.csv",index=False)
    np.savez_compressed(out/"decision_arrays.npz",observations=X,masks=M,feature_names=np.asarray(names))
    episode=pd.DataFrame(episodes)
    episode.to_csv(out/"tables/episode_metrics.csv",index=False)
    agg=[]
    for ck,g in data.groupby("checkpoint"):
        legal=g[g.legal_count>=2]
        unlock=g[g.irrigation_unlock]
        e=episode[episode.checkpoint==ck]
        agg.append(dict(checkpoint=int(ck),rows=len(g),choice_rows=len(legal),mean_yield=e.yield_kg_ha.mean(),mean_profit=e.simple_profit_score.mean(),
                        unique_signatures=e.action_signature.nunique(),mean_entropy=legal.entropy_normalized.mean(),mean_top=legal.top_probability.mean(),
                        near_tie_fraction=(legal.top_gap<.02).mean(),first_layer_saturation=legal.first_tanh_saturated.mean(),
                        critic_G_RMSE=float(np.sqrt(np.square(legal.G_minus_V).mean())),critic_G_bias=legal.G_minus_V.mean(),
                        unlock_rows=len(unlock),irrigate_on_unlock_rate=(unlock.irrigation>0).mean() if len(unlock) else np.nan,
                        legal_irrigation_days=int(g.irrigation_legal.sum()),irrigation_events=int((g.irrigation>0).sum())))
    pd.DataFrame(agg).to_csv(out/"tables/checkpoint_diagnosis.csv",index=False)
    sensitivity=offline_checks(out,data,X,M,names,models)
    mem=memory_probe(out,config,env_config)
    updates=pd.read_csv(ref.PPO_ROOT/"logs/218YCA_training_update_metrics.csv")
    slope=np.polyfit(updates.env_create_count,updates.rss_mb,1)[0]
    train_summary=dict(first_rss=float(updates.rss_mb.iloc[0]),last_rss=float(updates.rss_mb.iloc[-1]),rss_mb_per_env_created=float(slope),
                       max_cache=int(updates.peak_cached_envs.max()),max_coverage=int(updates.scenario_coverage.max()),
                       median_explained_variance=float(updates.explained_variance.median()),negative_explained_variance_fraction=float((updates.explained_variance<0).mean()),
                       median_kl=float(updates.approx_kl.median()),median_clip_fraction=float(updates.clip_fraction.median()))
    write_json(out/"diagnosis_result.json",dict(status="completed",smoke=args.smoke,checkpoints=agg,sensitivity=sensitivity,memory=mem,training_log_summary=train_summary,
          hypotheses_not_causal_proof=["Raw-input saturation may reduce state sensitivity", "Critic mismatch may weaken temporal credit assignment"],
          matched_saved_actions=True,reward_identity_passed=True,feature_names=names))
    print(json.dumps(dict(status="completed",out=str(out),checkpoints=agg,memory=mem,training=train_summary),indent=2),flush=True)


if __name__=="__main__":
    main()
