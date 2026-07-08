"""Production data update helper tests."""

import io

from quantx.production import data_update


def test_run_update_command_resolves_python_to_current_interpreter(monkeypatch, tmp_path):
    calls = {}

    class FakeProcess:
        def __init__(self, command, cwd, text, stdout, stderr, bufsize):
            calls["command"] = command
            calls["cwd"] = cwd
            calls["text"] = text
            calls["stdout"] = stdout
            calls["stderr"] = stderr
            calls["bufsize"] = bufsize
            self.stdout = io.StringIO("ok\n")
            self.stderr = io.StringIO("progress 1/1\n")
            self.returncode = 0

        def wait(self, timeout):
            calls["timeout"] = timeout
            return self.returncode

        def kill(self):
            calls["killed"] = True

    monkeypatch.setattr(data_update.subprocess, "Popen", FakeProcess)

    result = data_update.run_update_command(
        ["python", "-m", "quantx.tools.sync_daily_data"],
        cwd=tmp_path,
        timeout_seconds=123,
        stream_output=False,
    )

    assert result["ok"] is True
    assert calls["command"][0] == data_update.sys.executable
    assert calls["command"][1:] == ["-m", "quantx.tools.sync_daily_data"]
    assert calls["cwd"] == str(tmp_path)
    assert calls["timeout"] == 123
    assert result["stdout"] == "ok\n"
    assert result["stderr"] == "progress 1/1\n"


def test_run_update_command_reports_timeout(monkeypatch, tmp_path):
    calls = {}

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self.returncode = -9

        def wait(self, timeout=None):
            if not calls.get("waited"):
                calls["waited"] = True
                raise data_update.subprocess.TimeoutExpired(["cmd"], timeout)
            return self.returncode

        def kill(self):
            calls["killed"] = True

    monkeypatch.setattr(data_update.subprocess, "Popen", FakeProcess)

    result = data_update.run_update_command(["python", "-V"], cwd=tmp_path, timeout_seconds=1, stream_output=False)

    assert result["ok"] is False
    assert result["error_type"] == "timeout"
    assert calls["killed"] is True
