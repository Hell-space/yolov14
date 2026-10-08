"""Verify built wheel/sdist data scripts without executing downloads."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    expected = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted((root / "ultralytics/data/scripts").glob("*.sh"))
    }
    assert "ultralytics/data/scripts/get_imagenet.sh" in expected, "Source ImageNet script is missing"
    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if not wheels or not sdists:
        parser.error("Build both a wheel and sdist first with: python -m build")

    for path in wheels:
        with zipfile.ZipFile(path) as archive:
            for name, content in expected.items():
                assert name in archive.namelist(), f"{path.name} is missing {name}"
                assert archive.read(name) == content, f"{path.name} has changed {name}"
        print(f"PASS {path.name}: {len(expected)} data scripts match the checkout")

    for path in sdists:
        prefix = path.name.removesuffix(".tar.gz")
        with tarfile.open(path, "r:gz") as archive:
            for name, content in expected.items():
                member = f"{prefix}/{name}"
                assert member in archive.getnames(), f"{path.name} is missing {name}"
                stream = archive.extractfile(member)
                assert stream is not None, f"{path.name}: {name} is not a file"
                with stream:
                    assert stream.read() == content, f"{path.name} has changed {name}"
        print(f"PASS {path.name}: {len(expected)} data scripts match the checkout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
