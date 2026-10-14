"""Extend the existing corrected-weather replay to the completed historical seed7."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import hl_fq_8seed_cross_site.replay_fq2018_corrected_existing_seeds as replay
replay.OUT=ROOT/'results/version2_fq_seed7_corrected2018_063'
replay.config_and_output=lambda site,seed:(ROOT/'results/fq_seed7_completion_20261012/fq_seed7.json',ROOT/'benchmark_results/20261012fq7_fq_seed7_historical_completion')
code=inspect.getsource(replay.main).replace('range(7)','range(8)')
exec(compile(code,__file__,'exec'),replay.__dict__)
sys.argv=[__file__,'--seed','7'];replay.main()
