from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ghostlab.settings import Settings


@dataclass(frozen=True)
class LocalStore:
    root: Path
    data_dir: Path
    outputs_dir: Path
    artifacts_dir: Path
    logs_dir: Path

    @staticmethod
    def from_settings(repo_root: Path, s: Settings) -> "LocalStore":
        # resolve paths relative to repo root
        data_dir = (repo_root / s.local_data_dir).resolve()
        outputs_dir = (repo_root / s.output_dir).resolve()
        artifacts_dir = (repo_root / s.artifact_dir).resolve()
        logs_dir = (repo_root / s.log_dir).resolve()
        return LocalStore(
            root=repo_root.resolve(),
            data_dir=data_dir,
            outputs_dir=outputs_dir,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
        )

    def ensure_dirs(self) -> None:
        for p in [self.data_dir, self.outputs_dir, self.artifacts_dir, self.logs_dir]:
            p.mkdir(parents=True, exist_ok=True)

    # canonical subpaths
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def samples_dir(self) -> Path:
        return self.data_dir / "samples"
