"""Task-Embedding Parzen Shapley (the TRAC method lives in the `trac` package)."""

from .anchors import (
    IndexedRelabelledDataset,
    configure_determinism,
    cpu_state_dict,
    save_checkpoint,
    stable_int,
    train_anchor,
)
from .models import CNNMnist, CNNCifar, build_task_model
from .estimators import (
    antithetic_mc,
    balanced_mc,
    control_mc,
    exact_shapley,
    vanilla_mc,
)
from .games import pcc, rmse, stable_seed, table_utility
from .utility import (
    ClientStatistics,
    assemble_parzen_utility,
    client_local_statistics,
    make_parzen_utility,
)
from .encoders import DEFAULT_KERNEL_TEMPERATURE, Encoder, StubEncoder, load_encoder
from .embeddings import (
    EmbeddingCache,
    apply_overrides,
    extract_client_embeddings,
    load_embedding_cache,
    save_embedding_cache,
)

__all__ = [
    "ClientStatistics",
    "DEFAULT_KERNEL_TEMPERATURE",
    "EmbeddingCache",
    "Encoder",
    "StubEncoder",
    "antithetic_mc",
    "apply_overrides",
    "assemble_parzen_utility",
    "balanced_mc",
    "client_local_statistics",
    "configure_determinism",
    "control_mc",
    "cpu_state_dict",
    "exact_shapley",
    "extract_client_embeddings",
    "load_embedding_cache",
    "load_encoder",
    "make_parzen_utility",
    "pcc",
    "rmse",
    "save_embedding_cache",
    "stable_seed",
    "table_utility",
    "vanilla_mc",
]
