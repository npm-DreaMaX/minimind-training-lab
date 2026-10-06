"""Exact real-data equivalence across storage type changes and shard boundaries."""
import json,random,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lab.omni_data import StreamingOmniDataset,ReproducibleOmniDataset,EpochTaggedSampler,official
from lab.data import EpochBatchSampler

def identity_collate(batch): return batch

class OmniStorageContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); cls.temp=tempfile.TemporaryDirectory(prefix='omni_storage_test_')
        cls.raw=ROOT/'artifacts/omni_validation/t2a_first_row_group.parquet'
        cls.shards=Path(cls.temp.name)/'shards'
        subprocess.run([sys.executable,str(ROOT/'tools/repack_omni.py'),'--raw',str(cls.raw),'--out',str(cls.shards),
                        '--allow-fixture','--shard-rows','512'],check=True,stdout=subprocess.DEVNULL)
        tok=AutoTokenizer.from_pretrained(ROOT/'upstream/model')
        cls.baseline=official.OmniDataset(str(cls.raw),tok,max_length=1536)
        cls.streamed=StreamingOmniDataset(str(cls.shards),tok,max_length=1536)
        cls.streamed.table.cache_size=2

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_all_columns_survive_uint16_and_cross_shard_eviction(self):
        self.assertEqual(len(self.baseline),len(self.streamed))
        for row in [0,1000,2000,3701,511,512,1,1024,3000,32]:
            for column in self.baseline.table.column_names:
                self.assertEqual(self.baseline.table[column][row].as_py(),self.streamed.table[column][row].as_py(),(row,column))
        self.assertLessEqual(len(self.streamed.table.cache),2)

    def test_original_labels_delay_pattern_and_random_augmentation_unchanged(self):
        for row in [0,7,511,512,1000,3701]:
            values=[]
            for dataset in [self.baseline,self.streamed]:
                random.seed(6103+row); np.random.seed(6103+row); torch.manual_seed(6103+row)
                values.append(dataset[row])
            for before,after in zip(*values):
                if isinstance(before,torch.Tensor): self.assertTrue(torch.equal(before,after),(row,before.shape))
                else: self.assertEqual(before,after)
            ids,labels,audio,*_=values[1]
            self.assertEqual(tuple(ids.shape),(9,1535)); self.assertEqual(tuple(audio.shape),(8,1535))
            self.assertGreater(int((labels!=-100).sum()),0)
            valid=audio[audio!=-100]; self.assertGreater(len(valid),0)
            self.assertTrue(bool(((valid<2048)|(valid==2050)).all()))

    def test_augmentation_replay_survives_workers_and_resume_without_changing_main_rng(self):
        dataset=ReproducibleOmniDataset(str(self.shards),self.baseline.tokenizer,max_length=1536,
                                        rows=np.asarray([0,7,511,512,1000,3701]),seed=61)
        def collect(workers,start):
            loader=torch.utils.data.DataLoader(dataset,
                batch_sampler=EpochTaggedSampler(EpochBatchSampler(len(dataset),2,42,epoch=3,start=start),3),
                num_workers=workers,collate_fn=identity_collate,generator=torch.Generator().manual_seed(65))
            return [item for batch in loader for item in batch]
        random.seed(611); np.random.seed(612); torch.manual_seed(613)
        states=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
        uninterrupted=collect(0,0); replay=collect(2,2)
        for before,after in zip(uninterrupted[2:],replay):
            for x,y in zip(before,after):
                if isinstance(x,torch.Tensor): self.assertTrue(torch.equal(x,y))
                else: self.assertEqual(x,y)
        self.assertEqual(states[0],random.getstate())
        self.assertTrue(np.array_equal(states[1][1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(states[2],torch.get_rng_state()))

if __name__=='__main__': unittest.main(verbosity=2)
