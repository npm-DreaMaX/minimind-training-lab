"""Check the target alignment and weighted reduction, including empty audio heads."""
import types,unittest
import torch
import torch.nn.functional as F
from lab.omni_loss import omni_loss

class OmniLossContract(unittest.TestCase):
    def test_matches_explicit_original_objective_and_gradients(self):
        torch.set_num_threads(2); torch.manual_seed(61)
        logits=torch.randn(2,7,32,requires_grad=True)
        audio=[torch.randn(2,7,2112,requires_grad=True) for _ in range(8)]
        labels=torch.randint(0,32,(2,7)); labels[0,:3]=-100
        alabels=torch.randint(0,2048,(2,8,7)); alabels[:,:,:2]=-100; alabels[:,:,-1]=2050
        alabels[:,3,:]=-100
        result=types.SimpleNamespace(logits=logits,audio_logits=audio,aux_loss=torch.tensor(.01))
        actual=omni_loss(result,labels,alabels,diagnostics=True)
        expected=F.cross_entropy(logits.flatten(0,1),labels.flatten())+.01
        for i,a in enumerate(audio):
            target=alabels[:,i].flatten(); valid=target!=-100
            if valid.any():
                element=F.cross_entropy(a.flatten(0,1)[valid],target[valid],reduction='none')
                expected=expected+(element*(1+9*(target[valid]==2050))).sum()/valid.sum()/8
        torch.testing.assert_close(actual['loss'],expected)
        ga=torch.autograd.grad(actual['loss'],[logits,*audio],retain_graph=True)
        ge=torch.autograd.grad(expected,[logits,*audio],allow_unused=True)
        for x,y in zip(ga,ge):
            if y is None: self.assertEqual(float(x.abs().sum()),0.)
            else: torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-5)
        self.assertEqual(actual['audio_counts'].tolist(),[10,10,10,0,10,10,10,10])
        self.assertEqual(actual['stop_counts'].tolist(),[2,2,2,0,2,2,2,2])
        per=actual['per_sample']
        torch.testing.assert_close(per['text_nll_sum'].sum()/per['text_count'].sum(),actual['text_ce'])
        torch.testing.assert_close(per['audio_weighted_nll_sum'].sum(0)/per['audio_count'].sum(0).clamp_min(1),actual['audio_codebook_ce'])
        self.assertEqual(per['stop_count'].sum(0).tolist(),actual['stop_counts'].tolist())
        self.assertEqual(per['stop_correct'].sum(0).tolist(),actual['stop_correct'].tolist())
        self.assertTrue(all(not value.requires_grad for value in per.values()))
        # Cohort metrics must match evaluating those examples directly, even
        # when the other example contributes different valid-label counts.
        one=omni_loss(types.SimpleNamespace(logits=logits[1:],audio_logits=[x[1:] for x in audio],aux_loss=result.aux_loss),labels[1:],alabels[1:])
        torch.testing.assert_close(per['text_nll_sum'][1]/per['text_count'][1],one['text_ce'])
        torch.testing.assert_close(per['audio_weighted_nll_sum'][1]/per['audio_count'][1].clamp_min(1),one['audio_codebook_ce'])

    def test_text_only_still_has_finite_loss_and_audio_zero_gradients(self):
        text=torch.randn(1,4,32,requires_grad=True)
        audio=[torch.randn(1,4,2112,requires_grad=True) for _ in range(8)]
        labels=torch.tensor([[-100,3,2,-100]])
        result=types.SimpleNamespace(logits=text,audio_logits=audio,aux_loss=torch.tensor(0.))
        loss=omni_loss(result,labels,torch.full((1,8,4),-100))
        self.assertTrue(torch.isfinite(loss['loss'])); self.assertEqual(float(loss['audio_ce']),0.)
        loss['loss'].backward()
        for head in audio: self.assertEqual(float(head.grad.abs().sum()),0.)

if __name__=='__main__': unittest.main(verbosity=2)
