"""TRAC: Target-distribution Reweighted server Alignment for Contribution
measurement -- in-training client contribution scores for federated learning.

- `scoring`: pseudo-gradient math (cosines, EMA score, signed aggregation
  weights, class-balanced root loss weights).
- `fl_round`: one communication round, client/root pseudo-gradients and the
  accuracy evaluation.
- `models`: the LeNet-scale client networks.
- `data`: hyperparameters, datasets, stored partitions and the server root.
- `diagnostics`: contamination masks and the PCC/AUC evaluation helpers.

Shares the training primitives (`anchors.train_anchor`) and seeds
(`games.stable_seed`) with the `parzen_shapley` package.
"""

from .data import (
    TRAC_HYPERPARAMS,
    client_label_arrays,
    cover_other_classes,
    expected_k,
    fl_dataset_objects,
    load_partition,
    server_and_eval_indices,
)
from .diagnostics import (
    contaminated_from_overrides,
    contaminated_mask,
    flip_round,
    group_auc,
    label_entropy,
    low_entropy_clients,
    pcc_ceiling,
)
from .fl_round import (
    RoundOutput,
    evaluate_accuracy,
    local_pseudo_gradient,
    root_pseudo_gradient,
    run_round,
)
from .models import FLCifarNet, FLMnistNet, build_fl_model
from .scoring import (
    apply_update,
    class_balanced_weights,
    cosine_ema,
    cosine_similarity,
    normalize_pseudo_gradient,
    pseudo_gradient_norm,
    root_cosines,
    signed_weights,
    weighted_aggregate,
)

__all__ = [
    "FLCifarNet",
    "FLMnistNet",
    "RoundOutput",
    "TRAC_HYPERPARAMS",
    "apply_update",
    "build_fl_model",
    "class_balanced_weights",
    "client_label_arrays",
    "contaminated_from_overrides",
    "contaminated_mask",
    "cosine_ema",
    "cosine_similarity",
    "cover_other_classes",
    "evaluate_accuracy",
    "expected_k",
    "fl_dataset_objects",
    "flip_round",
    "group_auc",
    "label_entropy",
    "load_partition",
    "local_pseudo_gradient",
    "low_entropy_clients",
    "normalize_pseudo_gradient",
    "pcc_ceiling",
    "pseudo_gradient_norm",
    "root_cosines",
    "root_pseudo_gradient",
    "run_round",
    "server_and_eval_indices",
    "signed_weights",
    "weighted_aggregate",
]
