# test/test_types.py
import subprocess

def test_type_hints():
    result = subprocess.run(
        ["mypy", "--strict", "cilpy"],
        capture_output=True,
        text=True
    )

    # Always print output
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, "Type checking failed"
