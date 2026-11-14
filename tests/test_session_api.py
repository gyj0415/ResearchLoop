import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from researchloop import storage, server
from researchloop.engine import core


class SessionAPITests(unittest.TestCase):
    def test_pause_resume_and_fork_use_same_http_contract_as_ui(self):
        with tempfile.TemporaryDirectory(dir=storage.ROOT / '.cache') as temp:
            root = Path(temp)
            with patch.object(storage, 'RUNS', root/'runs'), patch.object(server, 'RUNS', root/'runs'), patch.object(storage, 'MEMORY', root/'memory'), patch.object(server, 'RESEARCH_ENGINE', None), patch.object(server.SERVICE, 'active', None):
                api = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
                worker = threading.Thread(target=api.serve_forever, daemon=True)
                worker.start()
                self.addCleanup(api.server_close)
                self.addCleanup(api.shutdown)
                engine = core.create({'question': '界面接口验收：暂停后继续读取材料', 'model_source': 'local', 'max_steps': 3})
                entered, release = threading.Event(), threading.Event()
                def first_decision():
                    entered.set()
                    release.wait(5)
                    return {'tool': 'inspect', 'args': {}}
                engine.decide = first_decision
                with server.LOCK:
                    server.launch_research(engine)
                self.assertTrue(entered.wait(5))
                def post(suffix, body=None, run_id=None):
                    request = urllib.request.Request(f'http://127.0.0.1:{api.server_port}/api/runs/{run_id or engine.run["id"]}/{suffix}',
                        data=json.dumps(body or {}).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(request, timeout=10) as response:
                        return json.load(response)
                def idle():
                    for _ in range(300):
                        if not server.SERVICE.active:
                            return
                        time.sleep(.01)
                    self.fail('Research worker did not release active slot')
                self.assertTrue(post('pause')['requested'])
                release.set()
                idle()
                self.assertEqual(storage.read_json(engine.folder/'run.json')['status'], 'paused')
                self.assertEqual(engine.run['steps'], [])
                def pause_again(current):
                    current.pause_requested.set()
                    return {'tool': 'inspect', 'args': {}}
                with patch.object(core.ResearchEngine, 'agent_step', pause_again):
                    resumed = post('resume')
                    self.assertEqual(resumed['id'], engine.run['id'])
                    idle()
                    saved = storage.read_json(engine.folder/'run.json')
                    node = next(item for item in saved['trace'] if item['kind'] == 'tool_call')
                    branch = post('fork', {'node_id': node['id'], 'question': '沿用该检查点继续分析'})
                    idle()
                self.assertNotEqual(branch['id'], engine.run['id'])
                child = storage.read_json(storage.RUNS/branch['id']/'run.json')
                self.assertEqual(child['status'], 'paused')
                self.assertEqual(child['lineage']['node_id'], node['id'])
                self.assertEqual(len(child['turns']), 1)
                self.assertEqual(saved['steps'][0]['tool'], 'inspect')
                self.assertEqual(storage.read_json(engine.folder/'run.json'), saved)


if __name__ == '__main__':
    unittest.main()
