"""An opt-in numeric experiment must leave the FP32 baseline unchanged."""
import importlib.util,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
class HybridPrecisionContract(unittest.TestCase):
    def test_default_matches_archived_fp32_outputs_and_all_gradients(self):
        torch.set_num_threads(2); torch.manual_seed(6103)
        old=load('hybrid_before_precision',ROOT/'artifacts/transfer_validation/hybrid_before_precision.py')
        new=load('hybrid_precision_candidate',ROOT/'models/02_hybrid_moe/src/model_hybrid.py')
        cfg=dict(hidden_size=32,num_hidden_layers=4,use_moe=True,vocab_size=64)
        reference=old.MiniMindForCausalLM(old.MiniMindConfig(**cfg))
        candidate=new.MiniMindForCausalLM(new.MiniMindConfig(**cfg))
        candidate.load_state_dict(reference.state_dict(),strict=True)
        ids=torch.randint(3,64,(1,11)); labels=ids.clone(); values=[]
        for model in [reference,candidate]:
            result=model(ids,labels=labels); values.append(result.logits.detach())
            (result.loss+result.aux_loss).backward()
        self.assertTrue(torch.equal(*values))
        for (name,p),(_,q) in zip(reference.named_parameters(),candidate.named_parameters()):
            self.assertTrue(torch.equal(p.grad,q.grad),name)

    def test_opt_in_dtypes_keep_gates_and_parameters_fp32(self):
        torch.set_num_threads(2)
        module=load('hybrid_precision_dtypes',ROOT/'models/02_hybrid_moe/src/model_hybrid.py')
        layer=module.GatedDeltaNet(module.MiniMindConfig(hidden_size=32,linear_precision='bfloat16'),0)
        observed={}
        for name in ['in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj']:
            def make_hook(key):
                def hook(_m,_i,out): observed[key]=out.dtype
                return hook
            getattr(layer,name).register_forward_hook(make_hook(name))
        x=torch.randn(1,11,32,requires_grad=True); result,_=layer(x)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(result).all())
        for name in ['in_proj_qkv','in_proj_z','out_proj']: self.assertEqual(observed[name],torch.bfloat16)
        for name in ['in_proj_a','in_proj_b']: self.assertEqual(observed[name],torch.float32)
        for p in layer.parameters():
            self.assertEqual(p.dtype,torch.float32); self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(result.dtype,torch.float32)

if __name__=='__main__': unittest.main(verbosity=2)
