from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class DiagnosticsRunContext:
    save_dir: str
    project_root: str
    artifact_stem: str
    dataset_name: str
    model_name: str
    seed: int
    fixed_train_loader: Any
    test_loader: Any
    total_batches: int
    num_train_samples: int
    num_epochs: int | None
    num_steps: int | None
    initial_best_acc: float = 0.0
    initial_best_epoch: int = 0
    checkpoint_saver: Any = None
    noisy_indices: Any = None
    true_labels: Any = None
    auxiliary_test_loaders: List[Any] = None