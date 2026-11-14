"""HTTP user controls for extra budgets, retry boundaries and permission resolution."""
import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from researchloop import storage,server
from researchloop.engine import core
from researchloop.sessions import SessionManager


class RecoveryAPITests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(dir=storage.ROOT/'.cache');self.addCleanup(temp.cleanup)
        root=Path(temp.name)
        self.manager=SessionManager(max_parallel=1)
        for target,key,value in [(storage,'RUNS',root/'runs'),(server,'RUNS',root/'runs'),(storage,'MEMORY',root/'memory'),
                                  (server,'SESSIONS',self.manager),(server.SERVICE,'active',None)]:
            scope=patch.object(target,key,value);scope.start();self.addCleanup(scope.stop)
        self.addCleanup(self.manager.close)
        self.api=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        threading.Thread(target=self.api.serve_forever,daemon=True).start()
        self.addCleanup(self.api.server_close);self.addCleanup(self.api.shutdown)
        self.engine=core.create({'question':'HTTP 预算与批准测试，无模型调用','max_steps':4})

    def request(self,suffix,body):
        url=f'http://127.0.0.1:{self.api.server_port}/api/runs/{self.engine.run["id"]}/{suffix}'
        req=urllib.request.Request(url,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=12) as response:return json.load(response)

    def idle(self):
        for _ in range(500):
            if not self.manager.busy():return
            time.sleep(.02)
        self.fail('Worker did not settle')

    @staticmethod
    def pause(engine):
        engine.pause_requested.set()
        return {'tool':'inspect','args':{}}

    def test_exhausted_run_can_continue_with_unlimited_budget_and_keeps_previous_report(self):
        self.engine.decide=lambda:{'tool':'finish','args':{'conclusion':'测试未取得研究证据，本轮明确未完成。','citations':[],'lesson':''}}
        self.engine.execute()
        first=(self.engine.folder/'report-t1.md').read_bytes()
        with patch.object(core.ResearchEngine,'agent_step',self.pause):
            result=self.request('recover',{'budget_enabled':False})
            self.idle()
        saved=storage.read_json(self.engine.folder/'run.json')
        self.assertEqual(result['id'],self.engine.run['id'])
        self.assertEqual(len(saved['turns']),2);self.assertEqual(saved['status'],'paused')
        self.assertIsNone(saved['token_limit']);self.assertIsNone(saved['seconds_limit']);self.assertIsNone(saved['max_steps'])
        self.assertEqual((self.engine.folder/'report-t1.md').read_bytes(),first)

    def test_paused_run_adds_to_spent_allowance_without_refund(self):
        self.engine.budget.record({'input_tokens':20,'output_tokens':10})
        self.engine.run['calls']=[{'input_tokens':20,'output_tokens':10}]
        self.engine.run['status']='paused';self.engine.save()
        with patch.object(core.ResearchEngine,'agent_step',self.pause):
            self.request('recover',{'budget_enabled':True,'token_limit':50000,'seconds_limit':None,'max_steps':8})
            self.idle()
        saved=storage.read_json(self.engine.folder/'run.json')
        self.assertEqual(saved['token_limit'],50030);self.assertEqual(saved['budget']['total_tokens'],30)
        self.assertEqual(len(saved['turns']),1)

    def test_retry_endpoint_creates_branch_and_keeps_parent(self):
        failed=self.engine._perform_tool({'tool':'read','args':{'asset_id':'not-there'}})
        parent=storage.read_json(self.engine.folder/'run.json')
        with patch.object(core.ResearchEngine,'agent_step',self.pause):
            child=self.request('retry',{'node_id':failed['node_id'],'budget_enabled':False})
            self.idle()
        self.assertNotEqual(child['id'],parent['id'])
        self.assertEqual(storage.read_json(self.engine.folder/'run.json'),parent)
        saved=storage.read_json(storage.RUNS/child['id']/'run.json')
        self.assertEqual(saved['lineage']['action'],'retry_before')
        self.assertIsNone(saved['token_limit']);self.assertEqual(saved['access_mode'],'approval')

    def test_approve_endpoint_unblocks_only_requested_operation(self):
        sent=False
        def decide():
            nonlocal sent
            if not sent:
                sent=True;return {'tool':'write_file','args':{'path':'approved.txt','content':'approved by HTTP fixture'}}
            self.engine.pause_requested.set()
            return {'tool':'inspect','args':{}}
        self.engine.decide=decide;self.manager.submit(self.engine)
        for _ in range(400):
            pending=[r for r in self.engine.run.get('approval_requests',[]) if r['status']=='pending']
            if pending:break
            time.sleep(.02)
        else:self.fail('No permission request')
        path=self.engine.folder/'workspace/approved.txt';self.assertFalse(path.exists())
        value=self.request('approve',{'request_id':pending[0]['id'],'decision':'once'})
        self.assertEqual(value['status'],'approved');self.idle();self.assertTrue(path.exists())
        self.assertEqual(self.engine.run['permission_grants'],[])


if __name__=='__main__':unittest.main()
