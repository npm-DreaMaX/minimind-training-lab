"""Exercise guards and real controller subprocess sequencing without a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lab.pipeline import check_completed, sha256


class PipelineContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root/'stage'
        (self.run/'checkpoints').mkdir(parents=True)
        self.config = dict(epochs=1,model='fixture')
        self.write('config.json',self.config)
        self.write('provenance.json',dict(total_steps=7,train_rows=56))
        self.write('status.json',dict(status='complete',step=7,trained_tokens=100,
                                     validation=dict(validation_ce=1.0,final=True)))
        self.write('checkpoints/best_validation.json',dict(step=7,validation_ce=1.0))
        (self.run/'checkpoints/best_validation.pth').write_bytes(b'fixture-not-weights')
        (self.run/'checkpoints/latest_resume.pt').write_bytes(b'fixture-not-weights')
        self.dep=dict(run_dir='stage',expected_steps=7,min_trained_tokens=100,config_contract=self.config,
                      provenance_contract=dict(train_rows=56))

    def tearDown(self):self.tmp.cleanup()
    def write(self,name,obj):(self.run/name).write_text(json.dumps(obj))
    def test_accept_complete_budget(self):self.assertEqual(check_completed(self.root,self.dep)['selection']['step'],7)
    def test_reject_smoke_completion(self):
        self.dep['expected_steps']=700
        with self.assertRaises(ValueError):check_completed(self.root,self.dep)
    def test_reject_short_data(self):
        self.write('provenance.json',dict(total_steps=7,train_rows=5))
        with self.assertRaises(ValueError):check_completed(self.root,self.dep)
    def test_reject_short_tokens(self):
        self.dep['min_trained_tokens']=101
        with self.assertRaises(ValueError):check_completed(self.root,self.dep)
    def test_reject_nan(self):
        self.write('checkpoints/best_validation.json',dict(step=7,validation_ce=float('nan')))
        with self.assertRaises(ValueError):check_completed(self.root,self.dep)
    def test_reject_untrained_selection(self):
        self.write('checkpoints/best_validation.json',dict(step=0,validation_ce=1.0))
        with self.assertRaises(ValueError):check_completed(self.root,self.dep)
    def test_reject_changed_recipe(self):
        self.write('config.json',dict(epochs=0,model='fixture'))
        with self.assertRaises(ValueError):check_completed(self.root,self.dep)
    def test_reject_missing_resume(self):
        (self.run/'checkpoints/latest_resume.pt').unlink()
        with self.assertRaises(ValueError):check_completed(self.root,self.dep)

    def controller_fixture(self):
        for path in ['tools/run_formal_pipeline.py','lab/live_status.py','lab/pipeline.py']:
            (self.root/path).parent.mkdir(exist_ok=True)
            shutil.copy2(ROOT/path,self.root/path)
        (self.root/'runs').mkdir()
        fake = '''import json, pathlib, sys
c=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())
r=pathlib.Path(c['run_dir']);(r/'checkpoints').mkdir(parents=True)
with open('executions.txt','a') as f:f.write(c['run_dir']+'\\n')
(r/'config.json').write_text(json.dumps(c))
(r/'provenance.json').write_text(json.dumps(dict(total_steps=7,train_rows=56)))
(r/'status.json').write_text(json.dumps(dict(status='complete',step=7,trained_tokens=100,validation=dict(validation_ce=1.,final=True))))
(r/'checkpoints/best_validation.json').write_text(json.dumps(dict(step=7,validation_ce=1.)))
(r/'checkpoints/best_validation.pth').write_bytes(b'fixture-not-weights')
(r/'checkpoints/latest_resume.pt').write_bytes(b'fixture-not-weights')
'''
        (self.root/'fake_train.py').write_text(fake)
        stages=[]
        for name in ['a','b']:
            cfg=dict(run_dir=name)
            (self.root/f'{name}.json').write_text(json.dumps(cfg))
            d={**self.dep,'run_dir':name,'config_contract':cfg}
            stages.append(dict(id=name,kind='train',module='fake_train',config=f'{name}.json',completion=d,dependencies=[]))
        plan=dict(controller_dir='runs/controller',fingerprints={'fake_train.py':sha256(self.root/'fake_train.py')},stages=stages,min_free_gib=0)
        (self.root/'plan.json').write_text(json.dumps(plan))
        return plan

    def invoke(self,*args):
        return subprocess.run([sys.executable,'-B','tools/run_formal_pipeline.py','--plan','plan.json',*args],
                              cwd=self.root,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=10)
    def test_sequence_and_restart_do_not_retrain_completed_stages(self):
        self.controller_fixture()
        first=self.invoke();self.assertEqual(first.returncode,0,first.stderr)
        second=self.invoke('--resume');self.assertEqual(second.returncode,0,second.stderr)
        self.assertEqual((self.root/'executions.txt').read_text(),'a\nb\n')
        self.assertEqual(json.loads((self.root/'runs/controller/status.json').read_text())['event'],'pipeline_budget_complete')
    def test_failed_dependency_prevents_any_training(self):
        plan=self.controller_fixture();plan['stages'][0]['dependencies']=[self.dep]
        (self.root/'plan.json').write_text(json.dumps(plan));self.write('status.json',dict(status='failed'))
        result=self.invoke();self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.root/'executions.txt').exists())
        self.assertEqual(json.loads((self.root/'runs/controller/status.json').read_text())['event'],'failed')
    def test_changed_source_prevents_any_training(self):
        self.controller_fixture();(self.root/'fake_train.py').write_text('raise Exception()')
        self.assertNotEqual(self.invoke().returncode,0)
        self.assertFalse((self.root/'executions.txt').exists())


if __name__=='__main__':unittest.main()
