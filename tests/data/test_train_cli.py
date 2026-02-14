import time
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.data.utils import run_cli, run_cli_popen

DESKTOP = Path.home() / "Desktop"
save_log = False


def test_train_cli():
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        result = run_cli(["preprocess"], cwd=tmp_path, timeout=0)  # 0 for no timeout

        if save_log is True:
            log_file = DESKTOP / "dequant_cli_preprocess_test.txt"
            with log_file.open("w", encoding="utf-8") as f:
                f.write(f"ARGS: {result.args}\n")
                f.write(f"RETURN CODE: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout or "<empty>\n")
                f.write("\nSTDERR:\n")
                f.write(result.stderr or "<empty>\n")

        assert result.returncode == 0, result.stderr

        # check for data directory
        data_dir = tmp_path / ".data"
        assert data_dir.exists(), "data directory was not created"

        # check if there is files
        files = list(data_dir.rglob("*"))
        assert len(files) > 0, "preprocessing did not generate HOV files"

        proc = run_cli_popen(["train"], cwd=tmp_path)

        triggered = False
        start = time.time()

        assert proc.stdout is not None

        try:

            for line in proc.stdout:
                print(line)

                if "Epoch" in line or "%" in line:
                    triggered = True
                    proc.terminate()
                    break

                if time.time() - start > 180:
                    break

        finally:

            proc.kill()
            proc.wait()

        assert triggered is True, "Training progress bar did not start"

        
