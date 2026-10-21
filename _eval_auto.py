"""临时驱动：以自动测试模式运行 eval_llm.py，绕开 stdin 编码问题。"""
import builtins

_orig_input = builtins.input


def _auto_input(prompt=""):
    if "[0] 自动测试" in str(prompt):
        return "0"
    return _orig_input(prompt)


builtins.input = _auto_input
import runpy
import sys

sys.argv = ["eval_llm.py", "--load_from", "./model", "--weight", "full_sft", "--device", "cuda"]
runpy.run_path("eval_llm.py", run_name="__main__")
