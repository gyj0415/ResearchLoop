"""Independent research workers with a bounded FIFO execution queue."""
import threading
import time
from collections import deque
from . import storage


class SessionManager:
    def __init__(self, max_parallel=None):
        self.max_parallel = max_parallel
        self.jobs = {}
        self.running = set()
        self.queue = deque()
        self.threads = set()
        self.closed = False

    def capacity(self):
        return self.max_parallel or storage.settings()["execution"]["max_parallel_sessions"]

    def busy(self, run_id=None):
        with storage.LOCK:
            return run_id in self.jobs if run_id else bool(self.jobs)

    def assert_idle(self, run_id):
        if self.busy(run_id):
            raise ValueError("该会话已经在运行或排队，不能重复启动")

    def approve(self,run_id,request_id,decision):
        with storage.LOCK:
            job=self.jobs.get(run_id)
            if not job or run_id not in self.running:
                raise ValueError('研究没有运行，批准请求已失效')
            return job['engine'].permission_gate.resolve(request_id,decision)

    def snapshot(self):
        with storage.LOCK:
            return {"active_runs": list(self.jobs), "running_runs": [key for key in self.jobs if key in self.running],
                    "queued_runs": list(self.queue), "max_parallel_sessions": self.capacity()}

    def submit(self, engine):
        with storage.LOCK:
            if self.closed:
                raise ValueError("服务正在关闭")
            identifier = engine.run["id"]
            self.assert_idle(identifier)
            engine.budget.suspend()
            engine.run.update(status="queued", queued_at=storage.now())
            engine.save()
            self.jobs[identifier] = {"engine": engine, "queued_clock": time.monotonic()}
            self.queue.append(identifier)
            self._pump()

    def _pump(self):
        while not self.closed and self.queue and len(self.running) < self.capacity():
            identifier = self.queue.popleft()
            job = self.jobs[identifier]
            engine = job["engine"]
            self.running.add(identifier)
            engine.run["queue_wait_seconds"] = round(engine.run.get("queue_wait_seconds", 0) + time.monotonic() - job["queued_clock"], 3)
            engine.run.update(status="running", last_started_at=storage.now())
            engine.budget.resume_clock()
            engine.save()
            thread = threading.Thread(target=self._work, args=(identifier, engine), daemon=True)
            self.threads.add(thread)
            try:
                thread.start()
            except Exception:
                self.threads.discard(thread)
                self.running.discard(identifier)
                self.jobs.pop(identifier, None)
                engine.run.update(status="failed", error="无法启动研究线程", finished_at=storage.now())
                engine.save()
                raise

    def _work(self, identifier, engine):
        try:
            engine.execute()
        except Exception as error:
            engine.run.update(status="failed", error=str(error), finished_at=storage.now())
            engine.save()
        finally:
            engine.budget.suspend()
            with storage.LOCK:
                if self.jobs.get(identifier, {}).get("engine") is engine:
                    self.jobs.pop(identifier, None)
                    self.running.discard(identifier)
                self.threads.discard(threading.current_thread())
                self._pump()

    def pause(self, run_id):
        with storage.LOCK:
            job = self.jobs.get(run_id)
            if not job:
                raise ValueError("该研究不在运行或排队中")
            engine = job["engine"]
            if run_id in self.running:
                if getattr(engine, "finishing", False):
                    return {"requested": False, "message": "研究正在收尾，结束后可以追问或从调用分支"}
                engine.pause_requested.set()
                engine.status_before_pause = engine.run["status"]
                engine.run["status"] = "pausing"
                engine.save()
                return {"requested": True, "message": "当前调用结束后暂停该研究，其他研究继续运行"}
            self.queue.remove(run_id)
            self.jobs.pop(run_id)
            engine.run.update(status="paused", paused_at=storage.now(), restore_budget=True)
            engine.event("paused", "研究在排队期间暂停，未启动新的调用", status="paused")
            return {"requested": True, "message": "排队研究已暂停"}

    def cancel(self, run_id):
        with storage.LOCK:
            job = self.jobs.get(run_id)
            if not job:
                raise ValueError("该研究不在运行或排队中")
            engine = job["engine"]
            if run_id in self.running:
                engine.cancelled.set()
                return {"requested": True, "message": "当前调用结束后停止该研究"}
            self.queue.remove(run_id)
            self.jobs.pop(run_id)
            engine.run.update(status="cancelled", finished_at=storage.now(), stop_reason="排队期间取消，未启动新的调用")
            engine.run["report"] = {"conclusion": "排队期间已取消，未启动新的研究调用。", "citations": [], "lesson": ""}
            if engine.run.get("turns"):
                engine.run["turns"][-1].update(finished_at=engine.run["finished_at"], conclusion=engine.run["report"]["conclusion"])
            engine.write_report()
            engine.event("cancelled", "排队研究已取消", status="cancelled")
            return {"requested": True, "message": "已取消排队研究"}

    def reconfigure(self):
        with storage.LOCK:
            self._pump()
            return self.snapshot()

    def close(self):
        """Used by tests: release any queued jobs and ask workers to stop."""
        with storage.LOCK:
            self.closed = True
            for engine_id in list(self.queue):
                self.cancel(engine_id)
            for job in self.jobs.values():
                job["engine"].cancelled.set()
            threads = list(self.threads)
        for thread in threads:
            thread.join(timeout=5)
