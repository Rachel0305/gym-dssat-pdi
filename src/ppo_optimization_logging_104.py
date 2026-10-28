"""Persist SB3's own post-train metrics; PPO math and RNG calls unchanged."""
import csv,os,math
from pathlib import Path
from sb3_contrib import MaskablePPO

FIELDS=['timesteps','n_updates','optimization_index','policy_gradient_loss','value_loss','loss','approx_kl','entropy_loss','entropy','clip_fraction','explained_variance','learning_rate','clip_range']
class LoggedMaskablePPO(MaskablePPO):
    def set_optimization_log(self,path):
        self.optimization_log_path=str(path)
        p=Path(path)
        assert not p.exists(),'Optimization log must not be overwritten'
        with p.open('x',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS).writeheader()
    def train(self):
        super().train()
        values=self.logger.name_to_value
        row=dict(timesteps=int(self.num_timesteps),n_updates=int(self._n_updates),optimization_index=int(self._n_updates//self.n_epochs))
        for name in FIELDS[3:]:
            if name=='entropy':continue
            value=values.get('train/'+name)
            row[name]=None if value is None else float(value)
        assert all(row[k] is not None and math.isfinite(row[k]) for k in ['policy_gradient_loss','value_loss','loss','approx_kl','entropy_loss']),row
        row['entropy']=-row['entropy_loss']
        with Path(self.optimization_log_path).open('a',newline='') as f:
            csv.DictWriter(f,fieldnames=FIELDS).writerow(row);f.flush();os.fsync(f.fileno())
