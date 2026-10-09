"""Port frozen FQA weather archive gate to the isolated HLA lowIC inputs."""

from pathlib import Path

root = Path(__file__).resolve().parents[2]
source = root / "results/fqa_wgen_multiyear_heldout_gate_045/run_gate.py"
target = Path(__file__).resolve().parent / "run_weather_gate.py"
if target.read_bytes() != source.read_bytes():
    raise RuntimeError("Starter does not match frozen FQA archive gate")
text = source.read_text(encoding="utf-8")
replacements = (
    ("src/051_fqa_originIC_site_transfer", "src/054_hla_lowIC_site_transfer"),
    ("results/fqa_wgen_ppo_smoke_044/run_smoke.py", "results/hla_wgen_8seed_053/run_ppo_smoke.py"),
    ("results/fqa_wgen_ppo_smoke_044/attempt_05/models/fqa_ppo_seed0_2k.zip", "results/hla_wgen_8seed_053/smoke_gpcc_raw_432_attempt02/models/hla_ppo_seed0_432.zip"),
    ("prompt_02/045_fqa_wgen_multiyear_heldout_archive_gate.md", "prompt_02/053_hla_wgen_8seed_resume.md"),
    ("range(2005, 2014)", "range(2004, 2014)"),
    ('"FQA"', '"HLA"'),
    ('"CNFQ.CLI"', '"CNHL.CLI"'),
    ('"CNFQ"', '"CNHL"'),
    ("fqa_045_", "hla_053_"),
)
for old, new in replacements:
    if old not in text:
        raise RuntimeError(f"Missing fragment: {old}")
    text = text.replace(old, new)
target.write_text(text, encoding="utf-8", newline="\n")
print(target.relative_to(root).as_posix())
