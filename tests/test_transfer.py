"""The whitelist must reject accidental omissions instead of silently resetting weights."""
import importlib,importlib.util,sys,types,unittest,warnings
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.transfer import transfer_ar_weights

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

class TransferContract(unittest.TestCase):
    def test_hybrid_shared_weights_and_missing_key_guard(self):
        torch.set_num_threads(2); torch.manual_seed(6)
        ar=load('test_transfer_ar',ROOT/'upstream/model/model_minimind.py')
        hy=load('test_transfer_hybrid',ROOT/'models/02_hybrid_moe/src/model_hybrid.py')
        cfg=dict(hidden_size=32,num_hidden_layers=4,vocab_size=64,use_moe=True)
        source=ar.MiniMindForCausalLM(ar.MiniMindConfig(**cfg))
        target=hy.MiniMindForCausalLM(hy.MiniMindConfig(**cfg))
        before={k:v.clone() for k,v in target.state_dict().items() if 'linear_attn.' in k}
        report=transfer_ar_weights(target,source.state_dict(),'hybrid')
        for key in report['copied_keys']: self.assertTrue(torch.equal(target.state_dict()[key],source.state_dict()[key]),key)
        for key,value in before.items(): self.assertTrue(torch.equal(value,target.state_dict()[key]),key)
        self.assertTrue(report['discarded_keys']); self.assertTrue(report['new_keys'])
        broken=dict(source.state_dict()); del broken['model.norm.weight']
        with self.assertRaises(ValueError): transfer_ar_weights(target,broken,'hybrid')

    def test_omni_thinker_logits_and_talker_block_copy(self):
        torch.set_num_threads(2); torch.manual_seed(6)
        package=types.ModuleType('transfer_omni_test'); package.__path__=[str(ROOT/'sources/minimind-o/model')]
        sys.modules[package.__name__]=package
        module=importlib.import_module(package.__name__+'.model_omni')
        cfg=dict(hidden_size=32,num_hidden_layers=2,talker_hidden_size=32,num_talker_hidden_layers=2,vocab_size=64,use_moe=True)
        source=module.MiniMindForCausalLM(module.MiniMindConfig(**cfg)).eval()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            target=module.MiniMindOmni(module.OmniConfig(**cfg),audio_encoder_path='/nonexistent',vision_model_path=None).eval()
        report=transfer_ar_weights(target,source.state_dict(),'omni')
        self.assertFalse(report['discarded_keys'])
        for entry in report['duplicated_from_thinker']:
            self.assertTrue(torch.equal(target.state_dict()[entry['target']],source.state_dict()[entry['source']]),entry)
        ids=torch.randint(3,64,(1,7))
        with torch.no_grad(): torch.testing.assert_close(target(ids).logits,source(ids).logits,atol=0,rtol=0)

if __name__=='__main__': unittest.main(verbosity=2)
