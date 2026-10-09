"""Create the isolated HLA 432-step PPO smoke from the frozen FQA smoke template."""

from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / "results/fqa_wgen_ppo_smoke_044/run_smoke.py"
target = Path(__file__).resolve().parent / "run_ppo_smoke.py"
if not target.exists():
    raise FileNotFoundError(target)
if target.read_bytes() != source.read_bytes():
    raise RuntimeError("053 smoke starter copy differs from frozen FQA template")

text = source.read_text(encoding="utf-8")
replacements = (
    ("src/051_fqa_originIC_site_transfer", "src/054_hla_lowIC_site_transfer"),
    ("run_051_00_fqa_originIC_expanded_action_maskableppo", "run_054_00_hla_lowIC_expanded_action_maskableppo"),
    ('parent / "attempt_05"', 'parent / "smoke_gpcc_raw_432"'),
    ("results/fqa_weather_resume_041/CNFQ.CLI", "results/hla_weather_enhancement_029/weather_fitting/gpcc_raw/CNHL.CLI"),
    ("prompt_02/044_fqa_wgen_ppo_2k_episode_archive_smoke.md", "prompt_02/053_hla_wgen_8seed_resume.md"),
    ("STEPS = 2000", "STEPS = 432"),
    ('"originIC"', '"lowIC"'),
    ("FQA", "HLA"),
    ("CNFQ", "CNHL"),
    ("fqa_ppo_seed0_2k.zip", "hla_ppo_seed0_432.zip"),
)
for before, after in replacements:
    if before not in text:
        raise RuntimeError(f"Template fragment missing: {before}")
    text = text.replace(before, after)
target.write_text(text, encoding="utf-8", newline="\n")
print(target.relative_to(root).as_posix())
