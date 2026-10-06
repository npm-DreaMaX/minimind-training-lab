"""Run with the isolated Omni environment; tests do not use the GPU."""
import copy,importlib,sys,types,unittest,warnings
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
package=types.ModuleType('official_omni_test'); package.__path__=[str(ROOT/'sources/minimind-o/model')]
sys.modules[package.__name__]=package
omni=importlib.import_module(package.__name__+'.model_omni')
from lab.checkpointing import checkpoint_blocks

class FrozenAudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.projection=torch.nn.Linear(560,512,bias=False); self.requires_grad_(False)
    def forward(self,features,lengths): return self.projection(features),lengths

class OmniContract(unittest.TestCase):
    def test_cached_logits_with_audio_prefix_and_speaker(self):
        torch.set_num_threads(2); torch.manual_seed(20261010)
        cfg=omni.OmniConfig(hidden_size=32,num_hidden_layers=2,talker_hidden_size=32,num_talker_hidden_layers=2,
                           vocab_size=64,use_moe=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model=omni.MiniMindOmni(cfg,audio_encoder_path='/nonexistent',vision_model_path=None).eval()
        encoder=FrozenAudioEncoder().eval(); object.__setattr__(model,'audio_encoder',encoder)
        calls=[]; encoder.register_forward_hook(lambda *_: calls.append(1))
        ids=torch.cat([torch.randint(0,2048,(1,8,15)),torch.randint(20,64,(1,1,15))],dim=1)
        ids[:,8,2:5]=16; ids[:,:8,0]=2051
        extra=dict(audio_inputs=torch.randn(1,3,560),audio_lens=torch.tensor([3]),spk_emb=torch.randn(1,192))
        with torch.inference_mode():
            full=model(ids,**extra)
            prefix=model(ids[:,:,:8],use_cache=True,**extra); cache=prefix.past_key_values
            text=[prefix.logits]; audio=[[head] for head in prefix.audio_logits]
            for token in ids[:,:,8:].split(1,dim=2):
                result=model(token,past_key_values=cache,use_cache=True,**extra); cache=result.past_key_values
                text.append(result.logits)
                for channel,head in zip(audio,result.audio_logits): channel.append(head)
            torch.testing.assert_close(full.logits,torch.cat(text,dim=1),atol=2e-6,rtol=2e-5)
            for expected,parts in zip(full.audio_logits,audio):
                torch.testing.assert_close(expected,torch.cat(parts,dim=1),atol=2e-6,rtol=2e-5)
            self.assertEqual(len(cache),4)
            self.assertTrue(all(k.shape[1]==15 and v.shape[1]==15 for k,v in cache))
        self.assertEqual(len(calls),2,'Audio encoder should run for full/prefill only, not every cached step')

    def test_frozen_backbone_keeps_audio_projector_gradient_with_recomputation(self):
        torch.set_num_threads(2); torch.manual_seed(61)
        cfg=omni.OmniConfig(hidden_size=32,num_hidden_layers=2,talker_hidden_size=32,num_talker_hidden_layers=2,
                           vocab_size=64,use_moe=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plain=omni.MiniMindOmni(cfg,audio_encoder_path='/nonexistent',vision_model_path=None).train()
        object.__setattr__(plain,'audio_encoder',FrozenAudioEncoder().eval())
        plain.requires_grad_(False); plain.audio_proj.requires_grad_(True)
        recomputed=copy.deepcopy(plain); checkpoint_blocks(list(recomputed.thinker.layers)+list(recomputed.talker.layers))
        ids=torch.randint(20,64,(1,12)); ids[:,2:5]=16
        fbank=torch.randn(1,3,560); labels=torch.randint(20,64,(1,12))
        losses=[]
        for model in [plain,recomputed]:
            result=model(ids,audio_inputs=fbank,audio_lens=torch.tensor([3]))
            loss=F.cross_entropy(result.logits.flatten(0,1),labels.flatten())
            losses.append(loss.detach()); loss.backward()
            self.assertTrue(all(p.grad is None for p in model.thinker.parameters()))
            self.assertTrue(all(p.grad is None for p in model.audio_encoder.parameters()))
            for p in model.audio_proj.parameters():
                self.assertIsNotNone(p.grad); self.assertGreater(float(p.grad.abs().sum()),0.)
        torch.testing.assert_close(losses[0],losses[1],atol=0,rtol=0)
        for a,b in zip(plain.audio_proj.parameters(),recomputed.audio_proj.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)

    def test_checkpointing_preserves_branched_thinker_talker_gradients(self):
        torch.set_num_threads(2); torch.manual_seed(6103)
        cfg=omni.OmniConfig(hidden_size=32,num_hidden_layers=2,talker_hidden_size=32,num_talker_hidden_layers=2,
                           vocab_size=64,use_moe=True,dropout=.1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plain=omni.MiniMindOmni(cfg,audio_encoder_path='/nonexistent',vision_model_path=None).train()
        recomputed=copy.deepcopy(plain)
        checkpoint_blocks(list(recomputed.thinker.layers)+list(recomputed.talker.layers))
        self.assertEqual(list(plain.state_dict()),list(recomputed.state_dict()))
        ids=torch.cat([torch.randint(0,2048,(1,8,11)),torch.randint(3,64,(1,1,11))],dim=1)
        text=torch.randint(3,64,(1,11)); audio=torch.randint(0,2048,(1,8,11)); losses=[]
        for model in [plain,recomputed]:
            torch.manual_seed(901); out=model(ids)
            loss=F.cross_entropy(out.logits.flatten(0,1),text.flatten())+out.aux_loss
            for layer,logits in enumerate(out.audio_logits):
                loss=loss+F.cross_entropy(logits.flatten(0,1),audio[:,layer].flatten())/8
            losses.append(loss.detach()); loss.backward()
        torch.testing.assert_close(losses[0],losses[1],atol=0,rtol=0)
        for (name,a),(_,b) in zip(plain.named_parameters(),recomputed.named_parameters()):
            self.assertIsNotNone(a.grad,name); self.assertIsNotNone(b.grad,name)
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5,msg=name)

if __name__=='__main__': unittest.main(verbosity=2)
