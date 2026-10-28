"""Project-local Double DQN with masks in selection and target evaluation."""
import torch
from torch.nn import functional as F
from strict_maskable_dqn_040 import StrictMaskableDQN
class StrictDoubleDQN(StrictMaskableDQN):
    def targets(self,batch):
        with torch.no_grad():
            mask=batch.next_masks.clone();mask[batch.dones.bool(),0]=True
            selected=self.online(batch.next_observations).masked_fill(~mask,-torch.inf).argmax(1)
            next_q=self.target(batch.next_observations).gather(1,selected[:,None]).squeeze(1)
            next_q=torch.where(batch.dones.bool(),torch.zeros_like(next_q),next_q)
            return batch.rewards+self.gamma*next_q
    def train_step(self):
        b=self.replay.sample(self.batch_size,self.rng,self.device);q=self.online(b.observations).gather(1,b.actions[:,None]).squeeze(1);target=self.targets(b)
        loss=F.smooth_l1_loss(q,target);assert torch.isfinite(loss)
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.online.parameters(),self.max_grad_norm);self.optimizer.step();self.optimizer_updates+=1
        return dict(loss=float(loss.detach()),mean_chosen_q=float(q.detach().mean()),mean_target=float(target.mean()),preclip_grad_norm=float(norm))
    def predict(self,obs,action_masks,deterministic=True):
        import numpy as np
        return np.asarray(self.select_action(obs,action_masks,0,deterministic=True)),None
def technical_test():
    import numpy as np
    m=StrictDoubleDQN(2,3,net_arch=[4],batch_size=2,buffer_size=8)
    with torch.no_grad():
        for network in [m.online,m.target]:
            for p in network.parameters():p.zero_()
        m.online.net[-1].bias.copy_(torch.tensor([10.,5.,999.]));m.target.net[-1].bias.copy_(torch.tensor([1.,100.,9999.]))
    assert m.select_action(np.zeros(2),np.array([True,True,False]),0,True)==0
    for terminal in [False,True]:m.add_transition(observation=np.zeros(2),action=0,reward=2.,next_observation=np.zeros(2),done=terminal,mask=np.array([True,True,False]),next_mask=np.zeros(3,dtype=bool) if terminal else np.array([True,True,False]))
    b=m.replay.sample(32,m.rng,m.device);t=m.targets(b);assert torch.allclose(t,torch.where(b.dones.bool(),torch.full_like(t,2),torch.full_like(t,2+m.gamma)))
    return dict(status='passed',online_select_target_evaluate=True,illegal_high_Q_excluded=True,terminal_no_bootstrap=True)
