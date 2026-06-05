# 后台任务运行器：在子线程中执行耗时操作（训练/回测等），
# 捕获 print() 与 logging 输出，供网页通过 SSE 实时滚动显示。

import io
import sys
import time
import uuid
import logging
import threading
import contextlib

logger = logging.getLogger(__name__)


class _LineEmitter(io.TextIOBase):
    """把写入的文本追加到任务的 lines 列表（线程安全：list.append 在 GIL 下原子）。"""

    def __init__(self, task):
        self._task = task

    def write(self, s):
        if s:
            self._task.lines.append(s)
        return len(s) if s else 0

    def flush(self):
        pass


class Task:
    def __init__(self, name: str):
        self.id      = uuid.uuid4().hex[:8]
        self.name    = name
        self.lines   = []           # 累积的输出片段
        self.status  = "running"    # running / done / error
        self.start   = time.time()
        self.elapsed = 0.0


class TaskManager:
    """同一时刻只允许一个重任务运行，避免 stdout 重定向相互干扰、资源争抢。"""

    def __init__(self):
        self._tasks = {}
        self._current = None
        self._lock = threading.Lock()

    def is_busy(self) -> bool:
        t = self._tasks.get(self._current)
        return t is not None and t.status == "running"

    def current(self):
        return self._tasks.get(self._current)

    def get(self, task_id: str):
        return self._tasks.get(task_id)

    def start(self, name: str, func) -> str | None:
        """启动一个后台任务；若已有任务在跑则返回 None。"""
        with self._lock:
            if self.is_busy():
                return None
            task = Task(name)
            self._tasks[task.id] = task
            self._current = task.id
        threading.Thread(target=self._run, args=(task, func), daemon=True).start()
        return task.id

    def _run(self, task: Task, func):
        emitter = _LineEmitter(task)
        handler = logging.StreamHandler(emitter)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            with contextlib.redirect_stdout(emitter), contextlib.redirect_stderr(emitter):
                func()
            task.status = "done"
            task.lines.append("\n✅ 任务完成\n")
        except Exception as e:
            task.status = "error"
            task.lines.append(f"\n❌ 任务失败: {e}\n")
            logger.exception("后台任务异常")
        finally:
            root.removeHandler(handler)
            task.elapsed = time.time() - task.start


# 全局单例
manager = TaskManager()
