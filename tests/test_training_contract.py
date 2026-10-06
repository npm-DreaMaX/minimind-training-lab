"""Scientific invariants; small CPU model is a unit test, never a training substitute."""
import copy,os,random,sys,unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset,SFTDataset
from lab.data import TokenCorpus,EpochBatchSampler
from lab.train import lr_at


class TrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_cached_pretrain_equals_official_prefix_and_labels(self):
        tok=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
        official=PretrainDataset(str(ROOT/'artifacts/pipeline_validation/pretrain_sample.jsonl'),tok,max_length=384)
        cached=TokenCorpus(ROOT/'artifacts/pipeline_validation/tokens','train',384,'pretrain')
        for i in [0,1,2,31,99,511,1000,len(cached)-1]:
            x,y=cached[i]; ox,oy=official[int(cached.rows[i])]
            self.assertTrue(torch.equal(x,ox)); self.assertTrue(torch.equal(y,oy))

    def test_shift_and_ignored_labels(self):
        torch.manual_seed(11)
        cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,vocab_size=64)
        model=MiniMindForCausalLM(cfg).eval()
        ids=torch.randint(3,64,(2,12)); labels=ids.clone(); labels[0,7:]=-100
        out=model(ids,labels=labels)
        manual=F.cross_entropy(out.logits[:,:-1].reshape(-1,64),labels[:,1:].reshape(-1),ignore_index=-100)
        torch.testing.assert_close(out.loss,manual)

    def test_sft_cache_matches_official_template_augmentation_and_labels(self):
        tok=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
        official=SFTDataset(str(ROOT/'artifacts/pipeline_validation/sft_sample.jsonl'),tok,max_length=1536)
        cached=TokenCorpus(ROOT/'artifacts/pipeline_validation/sft_tokens_v2','train',1536,'sft')
        positions={int(row):i for i,row in enumerate(cached.rows[:256])}
        random.seed(20261003)
        for row in range(256):
            ox,oy=official[row]
            if row in positions:
                x,y=cached[positions[row]]
                self.assertTrue(torch.equal(x,ox),row)
                self.assertTrue(torch.equal(y,oy),row)
                self.assertGreater(int((y[1:]!=-100).sum()),0)

    def test_token_weighted_accumulation_matches_dense_full_batch(self):
        torch.manual_seed(12)
        cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,vocab_size=64)
        full=MiniMindForCausalLM(cfg).eval(); micro=copy.deepcopy(full)
        ids=torch.randint(3,64,(2,12)); labels=ids.clone(); labels[0,4:]=-100
        full(ids,labels=labels).loss.backward()
        total=(labels[:,1:]!=-100).sum()
        for i in range(2):
            n=(labels[i:i+1,1:]!=-100).sum()
            (micro(ids[i:i+1],labels=labels[i:i+1]).loss*n/total).backward()
        for a,b in zip(full.parameters(),micro.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)

    def test_sampler_resume_has_no_skip_or_repeat(self):
        all_batches=list(EpochBatchSampler(103,8,42,epoch=2))
        resumed=list(EpochBatchSampler(103,8,42,epoch=2,start=24))
        self.assertEqual(all_batches[3:],resumed)
        self.assertEqual(len({i for b in all_batches for i in b}),103)

    def test_scheduler_position_independent_of_resume(self):
        values=[lr_at(i,100,10,3e-4) for i in range(100)]
        self.assertEqual(values[50:],[lr_at(i,100,10,3e-4) for i in range(50,100)])
        self.assertAlmostEqual(values[0],3e-5)
        self.assertAlmostEqual(values[9],3e-4)

    def test_recreating_loader_does_not_consume_global_rng(self):
        from torch.utils.data import DataLoader,TensorDataset
        dataset=TensorDataset(torch.arange(103))
        before=torch.get_rng_state().clone()
        for cursor in [0,24]:
            loader=DataLoader(dataset,batch_sampler=EpochBatchSampler(103,8,42,2,cursor),
                              generator=torch.Generator().manual_seed(2000047),num_workers=0)
            list(loader)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

if __name__=='__main__': unittest.main(verbosity=2)
