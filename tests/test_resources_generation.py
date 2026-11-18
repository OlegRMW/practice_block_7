import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from text_summary_block_7.resources import ResourceMeter
from text_summary_block_7.config import ROOT, load_config, select_judges
from text_summary_block_7.neural import ollama_generator, lora_generator


class ResourceTests(unittest.TestCase):
    def test_ollama_server_ram_and_gpu_exclude_other_processes(self):
        class Process:
            def __init__(self, pid=None):
                self.pid = os.getpid() if pid is None else pid
            def children(self, recursive=True):
                return [Process(101)] if self.pid == 100 else []
            def memory_info(self):
                return SimpleNamespace(rss={100: 10, 101: 20}.get(self.pid, 2) * 2**20)
        ps = SimpleNamespace(Process=Process, Error=OSError, CONN_LISTEN="listen", net_connections=lambda **k: [SimpleNamespace(pid=100, status="listen", laddr=SimpleNamespace(port=11434))])
        entries = [SimpleNamespace(pid=101, usedGpuMemory=3 * 2**20), SimpleNamespace(pid=9999, usedGpuMemory=1000 * 2**20)]
        nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None, nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda i: i, nvmlDeviceGetComputeRunningProcesses=lambda h: entries,
            nvmlDeviceGetGraphicsRunningProcesses=lambda h: entries, nvmlDeviceGetName=lambda h: "Test GPU")
        with patch.dict("sys.modules", {"psutil": ps, "pynvml": nvml}):
            with ResourceMeter("ollama", 10, "http://localhost:11434") as meter:
                pass
        self.assertEqual(meter.row["client_ram_peak_mib"], 2)
        self.assertEqual(meter.row["server_ram_peak_mib"], 30)
        self.assertEqual(meter.row["gpu_process_peak_mib"], 3)
        self.assertEqual(meter.row["gpu"], "Test GPU")

    def test_remote_server_memory_is_not_local_client_memory(self):
        with ResourceMeter("ollama", 10, "https://remote.example:11434") as meter:
            pass
        self.assertIsNone(meter.row["server_ram_peak_mib"])
        self.assertIsNone(meter.row["gpu_process_peak_mib"])
        self.assertIn("remote", meter.row["notes"])

    def test_both_backends_receive_64_and_report_actual_tokens(self):
        c = load_config(ROOT / "configs/qwen.yaml")
        calls = []
        response = SimpleNamespace(message=SimpleNamespace(content="summary"), eval_count=7, load_duration=10, eval_duration=20, prompt_eval_duration=10)
        client = SimpleNamespace(chat=lambda **kw: (calls.append(kw) or response))
        module = SimpleNamespace(Client=lambda **kw: client, ResponseError=RuntimeError)
        with patch.dict("sys.modules", {"ollama": module}):
            output = ollama_generator(c)("dialogue")
        self.assertEqual(calls[0]["options"]["num_predict"], 64)
        self.assertEqual(output.generated_tokens, 7)
        with tempfile.TemporaryDirectory(dir=ROOT / ".verification") as directory:
            Path(directory, "adapter_config.json").write_text("{}")
            class Inputs(dict):
                def to(self, device):
                    return self
            def generate(**kw):
                calls.append(kw)
                return [[11, 22, 33, 44]]
            model = SimpleNamespace(device="cuda:0", dtype="bf16", config=SimpleNamespace(), generate=generate)
            tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: Inputs(input_ids=SimpleNamespace(shape=(1, 2))), decode=lambda *a, **k: "summary")
            fast = SimpleNamespace(from_pretrained=lambda **kw: (model, tokenizer), for_inference=lambda m: None)
            c = load_config(ROOT / "configs/lora.yaml", named={"model.adapter_path": directory})
            with patch("text_summary_block_7.neural.fast_language_model", return_value=fast), patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=nullcontext)}):
                output = lora_generator(c)("dialogue")
            self.assertEqual(calls[-1]["max_new_tokens"], 64)
            self.assertEqual(output.generated_tokens, 2)

    def test_judge_selection_and_model_overrides(self):
        c = load_config(ROOT / "configs/lora.yaml", ["judge.max_tokens=2048", "judge.models.glm_flash.max_tokens=4096"])
        selected = select_judges(c)
        self.assertEqual(selected["gpt_oss"]["max_tokens"], 2048)
        self.assertEqual(selected["glm_flash"]["max_tokens"], 4096)
        self.assertEqual(list(select_judges(c, ["glm_flash"])), ["glm_flash"])
        with self.assertRaises(ValueError):
            select_judges(c, ["unknown"])
