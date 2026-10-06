import copy,sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from lab.checkpointing import checkpoint_blocks

class CheckpointContract(unittest.TestCase):
    def test_moe_task_and_aux_gradients_with_dropout(self):
        torch.set_num_threads(2); torch.manual_seed(70)
        cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                           vocab_size=64,use_moe=True,dropout=.1)
        plain=MiniMindForCausalLM(cfg).train(); recomputed=copy.deepcopy(plain)
        checkpoint_blocks(recomputed.model.layers)
        self.assertEqual(list(plain.state_dict()),list(recomputed.state_dict()))
        ids=torch.randint(3,64,(2,17)); labels=ids.clone(); labels[0,9:]=-100
        outputs=[]
        for model in [plain,recomputed]:
            torch.manual_seed(1234); out=model(ids,labels=labels)
            loss=out.loss+out.aux_loss; outputs.append(loss.detach()); loss.backward()
        torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
        for (name,a),(_,b) in zip(plain.named_parameters(),recomputed.named_parameters()):
            self.assertIsNotNone(a.grad,name); self.assertIsNotNone(b.grad,name)
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5,msg=name)

if __name__=='__main__': unittest.main(verbosity=2)
