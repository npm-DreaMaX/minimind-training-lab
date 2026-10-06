"""Move AdamW moment storage between updates; execute identical math on CUDA.

This trades about two moment-state transfers per optimizer update for lower
VRAM during all accumulation microbatches. Parameters and gradients stay CUDA.
Only non-capturable, non-fused, foreach=False AdamW is supported intentionally.
"""
import torch

class AdamMomentOffload:
    keys=('exp_avg','exp_avg_sq','max_exp_avg_sq')

    def __init__(self,optimizer,pin_memory=True):
        if not isinstance(optimizer,torch.optim.AdamW): raise TypeError('Only AdamW was audited')
        for group in optimizer.param_groups:
            if group.get('capturable') or group.get('fused') or group.get('foreach') is not False:
                raise ValueError('Moment offload requires non-capturable, non-fused, foreach=False AdamW')
            if any(p.device.type!='cuda' for p in group['params']): raise ValueError('Optimizer math must remain on CUDA')
        self.optimizer=optimizer; self.pin_memory=pin_memory; self.buffers={}; self.on_cpu=False

    def evict(self):
        """Copy exact moments to persistent host buffers and release CUDA refs."""
        moved=0
        for param,state in self.optimizer.state.items():
            for key in self.keys:
                if key not in state: continue
                value=state[key]
                if value.device.type!='cuda': raise RuntimeError('Unexpected state device before evict')
                slot=(param,key)
                if slot not in self.buffers:
                    self.buffers[slot]=torch.empty_like(value,device='cpu',pin_memory=self.pin_memory)
                host=self.buffers[slot]
                host.copy_(value,non_blocking=self.pin_memory)
                state[key]=host; moved+=value.numel()*value.element_size()
        # Saving or CPU diagnostics may read the host tensors immediately.
        torch.cuda.synchronize(); self.on_cpu=True
        return moved

    def restore(self):
        """Restore moments before optimizer.step; retain host buffers for reuse."""
        if not self.on_cpu: return 0
        moved=0
        for param,state in self.optimizer.state.items():
            for key in self.keys:
                if key not in state: continue
                value=state[key]
                if value.device.type!='cpu': raise RuntimeError('Unexpected state device before restore')
                state[key]=value.to(param.device,non_blocking=self.pin_memory)
                moved+=value.numel()*value.element_size()
        # Copies and subsequent optimizer ops use the same current stream.
        self.on_cpu=False
        return moved

    @property
    def host_bytes(self):
        return sum(v.numel()*v.element_size() for v in self.buffers.values())
