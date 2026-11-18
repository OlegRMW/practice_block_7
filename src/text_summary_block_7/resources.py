"""Sampled process memory and phase timers; unavailable metrics stay unavailable."""
import os
import platform
import threading
import time
from urllib.parse import urlparse


def cpu_name():
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


class ResourceMeter:
    def __init__(self, backend, interval=.2, host=None):
        self.backend, self.interval, self.host = backend, interval, host
        self.stop = threading.Event()
        self.client_peak = self.server_peak = None
        self.gpu_peak = None
        self.gpu_names = set()
        self.available_gpu_names = []
        self.notes = set()
        self.server_pids = set()
        self.observed_server_pids = set()
        self.nvml = None
        self.torch = None

    def __enter__(self):
        import psutil
        self.psutil = psutil
        self.cpu = cpu_name()
        self.process = psutil.Process()
        self.remote = False
        if self.backend == "ollama":
            parsed = urlparse(self.host)
            self.remote = parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            if self.remote:
                self.notes.add("remote Ollama: server process memory/hardware unavailable")
            else:
                try:
                    self.server_pids = {c.pid for c in psutil.net_connections(kind="tcp") if c.pid and c.status == psutil.CONN_LISTEN and c.laddr.port == (parsed.port or 11434)}
                except (psutil.Error, OSError):
                    self.notes.add("Ollama server PID inaccessible")
                if not self.server_pids:
                    self.notes.add("Ollama server PID not found")
        if self.backend == "torch":
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
        elif self.backend == "ollama" and not self.remote:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                for i in range(pynvml.nvmlDeviceGetCount()):
                    name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i))
                    self.available_gpu_names.append(name.decode() if isinstance(name, bytes) else name)
            except Exception:
                self.notes.add("NVML unavailable; process VRAM N/A")
        self.started = time.perf_counter()
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _processes(self, roots):
        found = {}
        for pid in roots:
            try:
                p = self.psutil.Process(pid)
                for proc in [p, *p.children(recursive=True)]:
                    found[proc.pid] = proc
            except self.psutil.Error:
                continue
        return found

    def _rss(self, processes):
        values = []
        for process in processes.values():
            try:
                values.append(process.memory_info().rss)
            except self.psutil.Error:
                self.notes.add("some process RSS inaccessible")
        return sum(values) / 2**20 if values else None

    def _sample(self):
        client = self._rss(self._processes({os.getpid()}))
        if client is not None:
            self.client_peak = max(self.client_peak or 0, client)
        server = self._processes(self.server_pids)
        self.observed_server_pids.update(server)
        value = self._rss(server)
        if value is not None:
            self.server_peak = max(self.server_peak or 0, value)
        if self.nvml and server:
            try:
                memory = 0
                observed = False
                for i in range(self.nvml.nvmlDeviceGetCount()):
                    handle = self.nvml.nvmlDeviceGetHandleByIndex(i)
                    entries = {}
                    for function in (self.nvml.nvmlDeviceGetComputeRunningProcesses, self.nvml.nvmlDeviceGetGraphicsRunningProcesses):
                        try:
                            for proc in function(handle):
                                if proc.pid in server and isinstance(proc.usedGpuMemory, int) and 0 <= proc.usedGpuMemory < 2**60:
                                    entries[proc.pid] = proc.usedGpuMemory
                        except Exception:
                            continue
                    if entries:
                        observed = True
                        memory += sum(entries.values())
                        name = self.nvml.nvmlDeviceGetName(handle)
                        self.gpu_names.add(name.decode() if isinstance(name, bytes) else name)
                if observed:
                    self.gpu_peak = max(self.gpu_peak or 0, memory / 2**20)
            except Exception:
                self.notes.add("GPU process memory inaccessible")

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self._sample()
            except Exception:
                self.notes.add("resource sampling partially unavailable")

    def __exit__(self, *exc):
        if self.torch and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join()
        self._sample()
        allocated = reserved = None
        if self.torch and self.torch.cuda.is_available():
            allocated = sum(self.torch.cuda.max_memory_allocated(i) for i in range(self.torch.cuda.device_count())) / 2**20
            reserved = sum(self.torch.cuda.max_memory_reserved(i) for i in range(self.torch.cuda.device_count())) / 2**20
            for i in range(self.torch.cuda.device_count()):
                if self.torch.cuda.max_memory_allocated(i):
                    self.gpu_names.add(self.torch.cuda.get_device_name(i))
        if self.backend == "ollama" and self.gpu_peak is None:
            self.notes.add("Ollama process VRAM unavailable (e.g. Windows WDDM); not total GPU memory")
        if self.backend == "ollama" and self.server_peak is None:
            self.notes.add("server RSS unavailable: remote server or inaccessible process")
        if self.client_peak is None:
            self.notes.add("Python RSS inaccessible")
        self.row = {"wall_seconds": self.seconds, "client_ram_peak_mib": self.client_peak,
                    "server_ram_peak_mib": self.server_peak, "gpu_process_peak_mib": self.gpu_peak,
                    "torch_allocated_peak_mib": allocated, "torch_reserved_peak_mib": reserved,
                    "cpu": self.cpu, "gpu": ", ".join(sorted(self.gpu_names)) or ("not used" if self.backend == "baseline" else "unknown"),
                    "available_gpu": ", ".join(self.available_gpu_names),
                    "client_pid": os.getpid(), "server_pids": ",".join(str(p) for p in sorted(self.observed_server_pids)),
                    "ram_scope": "sampled sum RSS: Python+children; Ollama server+children separately",
                    "sample_interval_seconds": self.interval, "notes": "; ".join(sorted(self.notes))}
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        return False
