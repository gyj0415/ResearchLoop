"""Concurrent HTTP jobs, queue controls and atomic shared method versions."""
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from researchloop import storage, server, db
from researchloop.engine import core
from researchloop.sessions import SessionManager


class ParallelSessionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=storage.ROOT/'.cache')
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        cfg=storage.settings()
        cfg.update(model_source='local',model='test-local',execution={'max_parallel_sessions':2})
        cfg['web']={'enabled':False};cfg['cloud']={'enabled':False,'api_key':''}
        for target,name,value in [(storage,'RUNS',self.root/'runs'),(storage,'MEMORY',self.root/'memory'),
                                  (server,'RUNS',self.root/'runs'),(server,'ROOT',self.root),
                                  (server.SERVICE,'active',None)]:
            scope=patch.object(target,name,value);scope.start();self.addCleanup(scope.stop)
        for target in (storage,server):
            scope=patch.object(target,'settings',return_value=cfg);scope.start();self.addCleanup(scope.stop)
        self.manager=SessionManager()
        scope=patch.object(server,'SESSIONS',self.manager);scope.start();self.addCleanup(scope.stop)
        self.gates={};self.counts={};self.started={}
        self.state_lock=threading.Lock()
        def decide(engine):
            identifier=engine.run['id']
            with self.state_lock:
                gate=self.gates.setdefault(identifier,threading.Event())
                self.started.setdefault(identifier,threading.Event()).set()
                self.counts[identifier]=self.counts.get(identifier,0)+1
            while not gate.wait(.02):
                if engine.cancelled.is_set():break
            if not engine.run.get('evidence_collected'):
                return {'tool':'inspect','args':{}}
            return {'tool':'finish','args':{'conclusion':'仅用于并行接口验收，没有真实科研证据，不形成结论。','citations':[],'lesson':''}}
        scope=patch.object(core.ResearchEngine,'agent_step',decide);scope.start();self.addCleanup(scope.stop)
        self.http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        threading.Thread(target=self.http.serve_forever,daemon=True).start()
        self.addCleanup(self.http.server_close);self.addCleanup(self.http.shutdown)
        def stop_workers():
            for gate in list(self.gates.values()):gate.set()
            self.manager.close()
        self.addCleanup(stop_workers)

    def request(self,path,body=None):
        request=urllib.request.Request(f'http://127.0.0.1:{self.http.server_port}'+path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type':'application/json'} if body is not None else {})
        try:
            with urllib.request.urlopen(request,timeout=10) as response:return response.status,json.load(response)
        except urllib.error.HTTPError as error:
            return error.code,json.load(error)

    def create(self,name):
        status,run=self.request('/api/research',{'question':f'并行接口验收：任务{name}的独立研究','model_source':'local',
                             'max_steps':6,'token_limit':40000,'seconds_limit':120,
                             'assets':[{'name':name+'.txt','content':name}]})
        self.assertEqual(status,202,run)
        return run['id']

    def wait(self,predicate):
        until=time.monotonic()+15
        while time.monotonic()<until:
            if predicate():return
            time.sleep(.01)
        self.fail(f'Condition did not become true: {self.manager.snapshot()}, calls={self.counts}')

    def read_run(self,identifier):
        return self.request('/api/runs/'+identifier)[1]

    def test_two_workers_overlap_third_queues_without_spending_time(self):
        a,b,c=self.create('A'),self.create('B'),self.create('C')
        self.wait(lambda:self.counts.get(a,0)>0 and self.counts.get(b,0)>0)
        _,state=self.request('/api/state')
        self.assertEqual(set(state['running_runs']),{a,b});self.assertEqual(state['queued_runs'],[c])
        self.assertNotIn(c,self.counts)
        before=self.read_run(c)['budget']['elapsed_seconds'];time.sleep(.1)
        self.assertEqual(self.read_run(c)['budget']['elapsed_seconds'],before)
        self.gates[a].set();self.wait(lambda:self.counts.get(c,0)>0)
        self.assertTrue(self.manager.busy(b));self.assertFalse(self.manager.busy(a))
        self.assertEqual(self.read_run(b)['status'],'running')
        self.assertEqual(self.read_run(c)['assets'][0]['name'],'C.txt')
        self.assertEqual((storage.RUNS/a/'inputs/a1').read_text(),'A')
        self.assertEqual((storage.RUNS/b/'inputs/a1').read_text(),'B')
        self.assertGreater(self.read_run(c)['queue_wait_seconds'],.05)

    def test_pause_waits_for_its_worker_and_resume_does_not_block_other_jobs(self):
        a,b,c=self.create('A'),self.create('B'),self.create('C')
        self.wait(lambda:self.counts.get(a,0)>0 and self.counts.get(b,0)>0)
        self.assertEqual(self.request(f'/api/runs/{a}/pause',{})[0],202)
        self.assertNotIn(c,self.counts)
        self.assertEqual(self.read_run(b)['status'],'running')
        self.gates[a].set();self.wait(lambda:not self.manager.busy(a) and self.counts.get(c,0)>0)
        self.assertEqual(self.read_run(a)['status'],'paused')
        self.gates[a].clear()
        self.assertEqual(self.request(f'/api/runs/{a}/resume',{})[0],202)
        self.assertIn(a,self.manager.snapshot()['queued_runs'])
        self.gates[b].set();self.wait(lambda:self.counts.get(a,0)>1)
        self.assertTrue(self.manager.busy(c))
        self.assertEqual(len(self.read_run(a)['turns']),1)

    def test_queued_pause_cancel_and_duplicate_resume_are_isolated(self):
        a,b,c=self.create('A'),self.create('B'),self.create('C')
        self.wait(lambda:self.counts.get(a,0)>0 and self.counts.get(b,0)>0)
        self.assertEqual(self.request(f'/api/runs/{c}/pause',{})[0],202)
        self.assertEqual(self.read_run(c)['status'],'paused');self.assertNotIn(c,self.counts)
        barrier=threading.Barrier(2)
        def resume():
            barrier.wait();return self.request(f'/api/runs/{c}/resume',{})[0]
        with ThreadPoolExecutor(2) as workers:codes=list(workers.map(lambda _:resume(),range(2)))
        self.assertEqual(sorted(codes),[202,400])
        self.assertEqual(self.request(f'/api/runs/{c}/cancel',{})[0],200)
        self.assertEqual(self.read_run(c)['status'],'cancelled');self.assertNotIn(c,self.counts)
        self.assertTrue((storage.RUNS/c/'report.md').exists())
        self.assertTrue(self.manager.busy(a));self.assertTrue(self.manager.busy(b))
        self.assertEqual(self.request(f'/api/runs/{b}/archive',{})[0],400)

    def test_queued_resume_preserves_already_spent_budget(self):
        self.create('A');self.create('B');c=self.create('C')
        engine=self.manager.jobs[c]['engine']
        item={'turn':'t1','source':'fixture','input_tokens':11,'output_tokens':7}
        engine.budget.record(item);engine.run['calls'].append(item);engine.save()
        self.assertEqual(self.request(f'/api/runs/{c}/pause',{})[0],202)
        spent=self.read_run(c)['budget']
        self.assertEqual(self.request(f'/api/runs/{c}/resume',{})[0],202)
        resumed=self.read_run(c)['budget']
        self.assertEqual(resumed['total_tokens'],18)
        self.assertEqual(resumed['model_calls'],1)
        self.assertAlmostEqual(spent['elapsed_seconds'],resumed['elapsed_seconds'],places=2)
        self.assertNotIn(c,self.counts)

    def test_fork_of_active_parent_does_not_pause_or_mutate_parent(self):
        engine=core.create({'question':'并行父会话分支验收任务','model_source':'local','max_steps':3})
        engine.controller_step('inspect',{})
        node=engine.run['trace'][0]['id']
        server.launch_research(engine);a=engine.run['id'];b=self.create('B')
        self.wait(lambda:self.counts.get(a,0)>0 and self.counts.get(b,0)>0)
        status,child=self.request(f'/api/runs/{a}/fork',{'node_id':node,'question':'分支独立进行验收'})
        self.assertEqual(status,202,child)
        self.assertIn(child['id'],self.manager.snapshot()['queued_runs'])
        self.assertEqual(self.read_run(a)['status'],'running')
        self.assertFalse(engine.pause_requested.is_set())
        self.assertEqual(self.read_run(child['id'])['lineage']['node_id'],node)

    def test_changing_capacity_does_not_cancel_workers_or_rewrite_other_settings(self):
        a,b,c=self.create('A'),self.create('B'),self.create('C')
        self.wait(lambda:self.counts.get(a,0)>0 and self.counts.get(b,0)>0)
        self.assertEqual(self.request('/api/settings',{})[0],400)
        status,state=self.request('/api/execution',{'max_parallel_sessions':3})
        self.assertEqual(status,200);self.wait(lambda:self.counts.get(c,0)>0)
        self.request('/api/execution',{'max_parallel_sessions':1})
        self.assertEqual(len(self.manager.snapshot()['running_runs']),3)
        self.assertEqual(storage.read_json(self.root/'config.json')['model'],'test-local')
        self.assertEqual(self.request('/api/execution',{'max_parallel_sessions':0})[0],400)

    def test_simultaneous_method_publication_has_unique_immutable_versions(self):
        barrier=threading.Barrier(4)
        def publish(index):
            barrier.wait()
            return core.publish({'domain':'research','status':'completed','id':f's{index}','question':f'q{index}',
                                 'report':{'lesson':f'method {index}'},'experiments':[]})
        with ThreadPoolExecutor(4) as workers:versions=list(workers.map(publish,range(4)))
        self.assertEqual({v['version'] for v in versions},{1,2,3,4})
        state=core.memory_state()
        self.assertEqual(state['active'],4)
        self.assertEqual([v['parent_version'] for v in state['versions']],[0,1,2,3])
        self.assertEqual(len(db.memory_list(scope='research')),4)


if __name__=='__main__':unittest.main()
