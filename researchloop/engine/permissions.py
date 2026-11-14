"""Session-scoped, explicit grants for file writes and command execution."""
import hashlib
import json
import uuid
from researchloop import storage
from researchloop.ollama import BudgetError
from . import history


class PermissionDenied(ValueError):
    pass


class PermissionGate:
    def __init__(self, owner):
        self.owner = owner
        self.once = set()

    def require(self, engine, category, details):
        owner = self.owner
        signature = hashlib.sha256(json.dumps([category, details], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with storage.LOCK:
            grants = owner.run.setdefault('permission_grants', [])
            if owner.run.get('access_mode', 'approval') == 'full' or category in grants:
                return
            requests = owner.run.setdefault('approval_requests', [])
            if any(r['signature'] == signature and r['status'] == 'denied' for r in requests):
                raise PermissionDenied('用户已拒绝该操作，不能重复执行；请选择其他方法。')
            request = {'id':'ap-'+uuid.uuid4().hex[:12], 'category':category, 'details':details,
                       'signature':signature, 'status':'pending', 'at':storage.now(),
                       'subagent':engine.run.get('subagent_id')}
            requests.append(request)
            owner.event('permission', '等待批准：'+category, status='pending', approval=request)
        try:
            while True:
                engine.guard_pause()
                if engine.cancelled.is_set():
                    raise PermissionDenied('研究已停止，未执行待批准操作。')
                if engine.seconds_left() <= 2:
                    raise BudgetError('等待批准时研究时间预算耗尽，未执行操作。')
                with storage.LOCK:
                    if request['status']=='pending' and category in owner.run.get('permission_grants',[]):
                        request.update(status='approved',decision='category',resolved_at=storage.now())
                        owner.save()
                    status = request['status']
                    if status == 'denied':
                        raise PermissionDenied('用户拒绝该操作，未执行。')
                    if status == 'approved':
                        if request['id'] in self.once:
                            self.once.remove(request['id'])
                        return
                engine.pause_requested.wait(.1)
        finally:
            with storage.LOCK:
                if request['status'] == 'pending':
                    request.update(status='cancelled', resolved_at=storage.now())
                    owner.save()

    def resolve(self, identifier, decision):
        if decision not in ('once','category','deny'):
            raise ValueError('请选择批准一次、批准本会话同类，或拒绝')
        with storage.LOCK:
            request = next((r for r in self.owner.run.get('approval_requests', []) if r['id'] == identifier), None)
            if not request or request['status'] != 'pending':
                raise ValueError('批准请求已失效或已处理')
            if self.owner.pause_requested.is_set() or self.owner.cancelled.is_set():
                raise ValueError('研究正在暂停或停止，不能批准新操作')
            request.update(status='denied' if decision == 'deny' else 'approved', decision=decision, resolved_at=storage.now())
            if decision == 'category':
                grants = self.owner.run.setdefault('permission_grants', [])
                if request['category'] not in grants:
                    grants.append(request['category'])
            elif decision == 'once':
                self.once.add(identifier)
            self.owner.event('permission', '批准处理：'+request['category'], status=request['status'], approval=request)
            return {'id':identifier, 'status':request['status'], 'decision':decision}
