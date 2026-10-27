"""Computation and replay backends behind the demo screens.

Every backend function reports through `progress(event)`, where `event` is a
plain dict with a `type` key (`progress`, `log`), and checks a
`threading.Event`-like `cancel` between stages. `run_job` evaluates one
(dataset, setting) with one method and puts the same events (plus
`dataset_start`, `result`, `done`, `cancelled`, `error`) on a queue for the
Tk thread.

Timing follows one record shape everywhere (`parzen_shapley.timing`):
`{"total_seconds", "stages", "rounds", "seeds", "source"}`; stored runs carry
it in results/*.json, real runs assemble it from the stage times the src
functions report, and `gt_seconds` is the exact-Shapley wall clock from the
manifest.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from . import config, texts


@dataclass
class Result:
    method: str                 # "parzen" | "trac"
    dataset: str
    key: str
    scenario: str               # "mislabel_4"
    label: str                  # "mislabel-4" (combobox value)
    n_players: int
    scores: list                # Parzen phi or TRAC cos_ema
    gt: list                    # ground_truth_group_shapley_phi
    contaminated: list          # bool per client
    pcc: float
    pcc_history: list           # TRAC: per-round PCC; Parzen: [pcc]
    rounds: int | None
    accuracy: float | None
    elapsed: float              # wall clock of this demo job (incl. replay animation)
    source: str                 # "real" | "replay" | "cache"
    timing: dict | None = None  # unified measurement-time record (see module docstring)
    gt_seconds: float | None = None  # exact-Shapley wall clock of this setting

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Result":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


class Cancelled(Exception):
    pass


def _emit(progress, **event) -> None:
    if progress is not None:
        progress(event)


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _sleep(seconds: float, cancel) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        _check(cancel)
        time.sleep(min(0.05, max(0.0, end - time.perf_counter())))


# --------------------------------------------------------------------------
# Setting catalog and stored results
# --------------------------------------------------------------------------

_CATALOG = None
_GT_ROWS = None
_TRAC = None


def label_of(setting) -> str:
    from parzen_shapley.pipeline import scenario

    return scenario(setting).replace("_", "-")


def catalog() -> dict:
    """dataset -> {label: setting} over the 24 non-iid settings, in the
    manifest's order (mislabel-k ascending, then noniid-k ascending)."""

    global _CATALOG
    if _CATALOG is None:
        from parzen_shapley.pipeline import selected_settings

        manifest = json.loads(config.MANIFEST.read_text())
        out = {dataset: {} for dataset in config.DATASETS}
        for setting in selected_settings(manifest):
            label = label_of(setting)
            if label == "iid":
                continue
            out[setting["dataset"]][label] = setting
        _CATALOG = out
    return _CATALOG


def labels_for(dataset: str) -> list:
    return list(catalog()[dataset])


def setting_for(dataset: str, label: str) -> dict:
    return catalog()[dataset][label]


def setting_by_key(key: str) -> dict:
    for per_dataset in catalog().values():
        for setting in per_dataset.values():
            if setting["key"] == key:
                return setting
    raise KeyError(key)


def _gt_rows() -> dict:
    global _GT_ROWS
    if _GT_ROWS is None:
        _GT_ROWS = {
            row["key"]: row for row in json.loads(config.GT_JSON.read_text())["rows"]
        }
    return _GT_ROWS


def ground_truth(key: str) -> np.ndarray:
    return np.asarray(_gt_rows()[key]["ground_truth_group_shapley_phi"], dtype=np.float64)


def stored_parzen_phi(key: str) -> np.ndarray:
    return np.asarray(_gt_rows()[key]["prediction_phi"], dtype=np.float64)


def gt_seconds(key: str) -> float | None:
    """Exact-Shapley (all-coalition retraining) wall clock from the manifest."""

    timing = setting_by_key(key).get("timing") or {}
    return timing.get("total_seconds")


def stored_parzen_timing(key: str) -> dict:
    row = _gt_rows()[key]
    if row.get("timing"):
        return dict(row["timing"])
    stages = {k: float(v) for k, v in row["timing_seconds"].items() if k != "exact_parzen_shapley"}
    return {"total_seconds": float(sum(stages.values())), "stages": stages, "rounds": None,
            "seeds": 1, "source": "timing_seconds without exact_parzen_shapley"}


def _trac_file() -> dict:
    global _TRAC
    if _TRAC is None:
        data = json.loads(config.TRAC_JSON.read_text())
        _TRAC = {"rows": data["rows"],
                 "summary": {entry["key"]: entry for entry in data.get("summary", [])}}
    return _TRAC


def stored_trac_row(key: str, seed_index: int = 0) -> dict:
    for row in _trac_file()["rows"]:
        if row["key"] == key and row["seed_index"] == seed_index:
            return row
    raise KeyError(f"no stored TRAC run for {key} seed {seed_index}")


def stored_trac_timing(key: str, rounds: int) -> dict:
    """Mean wall clock of the stored TRAC runs, scaled linearly when the replay
    stops before the stored run's last round (CIFAR-10: 40 stored, 30 replayed)."""

    summary = _trac_file()["summary"].get(key) or {}
    timing = summary.get("timing")
    if not timing:
        seconds = [row["wall_clock_seconds"] for row in _trac_file()["rows"] if row["key"] == key]
        if not seconds:
            raise KeyError(f"no stored TRAC run for {key}")
        mean = float(np.mean(seconds))
        timing = {"total_seconds": mean, "stages": {"fl_run": mean},
                  "rounds": int(stored_trac_row(key)["rounds"]), "seeds": len(seconds),
                  "source": f"mean wall_clock_seconds over {len(seconds)} seeds"}
    timing = dict(timing)
    stored_rounds = int(timing.get("rounds") or rounds)
    if stored_rounds != rounds:
        total = float(timing["total_seconds"]) * rounds / stored_rounds
        timing.update(total_seconds=total, stages={"fl_run": total}, rounds=rounds,
                      source=f"{timing['source']}, scaled {stored_rounds}->{rounds} rounds")
    return timing


def _timing_from_elapsed(result: "Result") -> dict:
    return {"total_seconds": float(result.elapsed), "stages": {}, "rounds": result.rounds,
            "seeds": 1, "source": "elapsed (legacy cache)"}


def stored_trac_trajectory(key: str, seed_index: int = 0):
    directory = config.TRAC_TRAJ_DIR
    for name in (f"{key}__epoch_gamma__seed{seed_index}.json", f"{key}__seed{seed_index}.json"):
        path = directory / name
        if path.is_file():
            return json.loads(path.read_text())["trajectory"]
    return None


def contamination(key: str, n_players: int | None = None) -> list:
    """Which clients are contaminated, from the stored TRAC run of the same
    key (identical partition for both methods). Unknown -> all False."""

    try:
        return [bool(x) for x in stored_trac_row(key)["contaminated"]]
    except (KeyError, FileNotFoundError):
        return [False] * int(n_players or 0)


def _pcc(a, b) -> float:
    from parzen_shapley import pcc

    return float(pcc(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def replay_parzen(key: str, progress=None, cancel=None,
                  stage_delay: float | None = None) -> Result:
    delay = config.REPLAY_STAGE_DELAY if stage_delay is None else stage_delay
    started = time.perf_counter()
    setting = setting_by_key(key)
    gt = ground_truth(key)
    phi = stored_parzen_phi(key)
    _emit(progress, type="log", text=texts.LOG_REPLAY)
    stages = texts.PARZEN_STAGES
    for step, stage in enumerate(stages, start=1):
        _check(cancel)
        _emit(progress, type="progress", value=step, max=len(stages),
              text=texts.STATUS_STAGE.format(stage=stage, step=step, total=len(stages)))
        _sleep(delay, cancel)
    pcc = _pcc(phi, gt)
    return Result(
        method="parzen", dataset=setting["dataset"], key=key,
        scenario=label_of(setting).replace("-", "_"), label=label_of(setting),
        n_players=len(gt), scores=phi.tolist(), gt=gt.tolist(),
        contaminated=contamination(key, len(gt)), pcc=pcc, pcc_history=[pcc],
        rounds=None, accuracy=None, elapsed=time.perf_counter() - started,
        source="replay",
        timing=stored_parzen_timing(key), gt_seconds=gt_seconds(key),
    )


def replay_trac(key: str, progress=None, cancel=None,
                round_delay: float | None = None) -> Result:
    delay = config.REPLAY_ROUND_DELAY if round_delay is None else round_delay
    started = time.perf_counter()
    setting = setting_by_key(key)
    gt = ground_truth(key)
    row = stored_trac_row(key)
    rounds = min(int(row["rounds"]), config.FULL_TRAC_ROUNDS[setting["dataset"]])
    trajectory = stored_trac_trajectory(key)
    _emit(progress, type="log", text=texts.LOG_REPLAY)
    # The stored run may be longer than the unified round count (CIFAR-10 was
    # run for 40): replay stops at `rounds` and reports that round's scores.
    truncated = int(row["rounds"]) > rounds
    scores = list(row["final_cos_ema"])
    pcc_final = float(row["pcc_cos_ema_vs_ground_truth"])
    accuracy = float(row["final_accuracy"])
    if trajectory:
        trajectory = trajectory[:rounds]
        if truncated:
            scores = list(trajectory[-1]["cos_ema"])
            pcc_final = float(trajectory[-1]["pcc_cos_ema_vs_gt"])
            accuracy = None  # only measured at the end of the stored run
        history = []
        for entry in trajectory:
            _check(cancel)
            history.append(float(entry["pcc_cos_ema_vs_gt"]))
            t = int(entry["round"]) + 1
            _emit(progress, type="progress", value=t, max=rounds,
                  text=texts.STATUS_ROUND.format(round=t, rounds=rounds, pcc=history[-1]))
            _sleep(delay, cancel)
    else:
        if truncated:
            raise FileNotFoundError(f"no stored trajectory to truncate for {key}")
        history = [pcc_final]
        _emit(progress, type="progress", value=rounds, max=rounds,
              text=texts.STATUS_ROUND.format(round=rounds, rounds=rounds, pcc=history[-1]))
    return Result(
        method="trac", dataset=setting["dataset"], key=key,
        scenario=label_of(setting).replace("-", "_"), label=label_of(setting),
        n_players=len(gt), scores=scores, gt=gt.tolist(),
        contaminated=[bool(x) for x in row["contaminated"]],
        pcc=pcc_final, pcc_history=history,
        rounds=rounds, accuracy=accuracy,
        elapsed=time.perf_counter() - started, source="replay",
        timing=stored_trac_timing(key, rounds), gt_seconds=gt_seconds(key),
    )


# --------------------------------------------------------------------------
# Real computation
# --------------------------------------------------------------------------


def ensure_anchor(setting: dict, train_dataset, device: str, progress=None, timer=None) -> Path:
    """Train and save the frozen task-model anchor for `setting` if missing.
    The stage is `anchor_training_or_load` (near zero when the anchor exists)."""

    from parzen_shapley.pipeline import train_setting_anchor

    path = config.ANCHOR_DIR / f"{setting['key']}.pt"
    with (timer.stage("anchor_training_or_load") if timer is not None else nullcontext()):
        if path.is_file():
            _emit(progress, type="log", text=texts.LOG_ANCHOR_FOUND)
            return path
        _emit(progress, type="log", text=texts.LOG_ANCHOR_TRAIN)
        train_setting_anchor(setting, train_dataset, path, device=device, seed=0)
    return path


def download_datasets(datasets=config.DATASETS) -> None:
    """Fetch MNIST / Fashion-MNIST / CIFAR-10 into data/ via torchvision.
    Only needed for '실제 계산' mode; replay reads results/ only."""
    from parzen_shapley.pipeline import dataset_objects

    for dataset in datasets:
        print(f"[{dataset}] downloading to {config.DATA_ROOT / dataset} ...", flush=True)
        dataset_objects(dataset, config.DATA_ROOT, download=True)
    print("done.")


def compute_parzen(key: str, device: str, progress=None, cancel=None) -> Result:
    import torch
    from types import SimpleNamespace

    from parzen_shapley import StageTimer, balanced_mc, stable_seed
    from parzen_shapley.pipeline import dataset_objects, prepare_setting_utility

    started = time.perf_counter()
    timer = StageTimer()
    setting = setting_by_key(key)
    dataset = setting["dataset"]
    gt = ground_truth(key)
    stages = texts.PARZEN_STAGES
    train_dataset, test_dataset = dataset_objects(dataset, config.DATA_ROOT, download=True)

    _check(cancel)
    _emit(progress, type="progress", value=0, max=len(stages),
          text=texts.STATUS_STAGE.format(stage=stages[0], step=1, total=len(stages)))
    ensure_anchor(setting, train_dataset, device, progress, timer)

    _check(cancel)
    _emit(progress, type="progress", value=1, max=len(stages),
          text=texts.STATUS_STAGE.format(stage=stages[1], step=2, total=len(stages)))
    args = SimpleNamespace(
        encoder="anchor", reference_dir=str(config.REFERENCE_DIR),
        anchor_dir=str(config.ANCHOR_DIR), seed=0, validation_size=2000,
        feature_batch_size=256, parzen_batch_size=512, kernel_temperature=20.0,
    )
    utility, metadata = prepare_setting_utility(
        setting, train_dataset, test_dataset, args, torch.device(device)
    )
    n_players = int(metadata["n_players"])
    for name, seconds in metadata.get("timing", {}).items():
        timer.add(name, seconds)

    _check(cancel)
    _emit(progress, type="progress", value=2, max=len(stages),
          text=texts.STATUS_STAGE.format(stage=stages[2], step=3, total=len(stages)))
    estimate = balanced_mc(
        utility, n_players, paths=2 * n_players * 8,
        seed=stable_seed(key, 0), control=False,
    )
    phi = np.asarray(estimate["phi"], dtype=np.float64)
    timer.add("balanced_estimation", estimate.get("seconds") or 0.0)
    _emit(progress, type="progress", value=3, max=len(stages), text="")
    pcc = _pcc(phi, gt)
    return Result(
        method="parzen", dataset=dataset, key=key,
        scenario=label_of(setting).replace("-", "_"), label=label_of(setting),
        n_players=n_players, scores=phi.tolist(), gt=gt.tolist(),
        contaminated=contamination(key, n_players), pcc=pcc, pcc_history=[pcc],
        rounds=None, accuracy=None, elapsed=time.perf_counter() - started,
        source="real",
        timing=timer.to_dict(rounds=None, seeds=1, source="measured"),
        gt_seconds=gt_seconds(key),
    )


def compute_trac(key: str, rounds: int, device: str, progress=None, cancel=None) -> Result:
    """The round loop of scripts/run_trac_sweep.py::run_one with a per-round
    progress event and cooperative cancellation."""

    import torch

    from parzen_shapley import StageTimer, configure_determinism, cpu_state_dict, stable_seed
    from trac import (
        TRAC_HYPERPARAMS,
        build_fl_model,
        class_balanced_weights,
        client_label_arrays,
        contaminated_mask,
        evaluate_accuracy,
        expected_k,
        fl_dataset_objects,
        load_partition,
        run_round,
        server_and_eval_indices,
    )

    started = time.perf_counter()
    timer = StageTimer()
    setting = setting_by_key(key)
    dataset = setting["dataset"]
    gt = ground_truth(key)
    hyperparams = TRAC_HYPERPARAMS[dataset]
    train_dataset, test_dataset = fl_dataset_objects(dataset, config.DATA_ROOT, download=True)

    def model_factory():
        return build_fl_model(dataset)

    with timer.stage("setup"):
        server_indices, _eval = server_and_eval_indices(
            test_dataset, validation_size=2000, server_pc=128, seed=0
        )
        root_labels = np.asarray(test_dataset.targets)[server_indices]
        class_weights = class_balanced_weights(root_labels, len(test_dataset.classes)).tolist()
        client_indices, overrides, _sizes = load_partition(config.REFERENCE_DIR, key)
        n_players = len(client_indices)
        contaminated = contaminated_mask(
            client_indices, overrides, client_label_arrays(train_dataset, client_indices),
            expected_k=expected_k(setting),
        )
        master_seed = stable_seed(key, 0)
        configure_determinism(master_seed)
        global_state = cpu_state_dict(model_factory())
    cos_ema = np.zeros(n_players, dtype=np.float64)
    history = []
    for t in range(rounds):
        _check(cancel)
        output = run_round(
            global_state, cos_ema, t,
            model_factory=model_factory, train_dataset=train_dataset,
            server_dataset=test_dataset, client_indices=client_indices,
            server_indices=server_indices, overrides=overrides,
            local_epochs=hyperparams["local_epochs"],
            batch_size=hyperparams["batch_size"],
            learning_rate=hyperparams["learning_rate"] * hyperparams["lr_decay_gamma"] ** t,
            gamma_target=hyperparams["gamma_target"], alpha=hyperparams["alpha"],
            server_class_weights=class_weights, device=torch.device(device),
            key=key, master_seed=master_seed,
        )
        global_state, cos_ema = output.global_state, output.cos_ema
        timer.add("fl_rounds", output.seconds)
        history.append(_pcc(cos_ema, gt))
        _emit(progress, type="progress", value=t + 1, max=rounds,
              text=texts.STATUS_ROUND.format(round=t + 1, rounds=rounds, pcc=history[-1]))

    with timer.stage("final_evaluation"):
        accuracy = evaluate_accuracy(
            global_state, model_factory, test_dataset, list(range(len(test_dataset))),
            batch_size=256, device=torch.device(device),
        )
    return Result(
        method="trac", dataset=dataset, key=key,
        scenario=label_of(setting).replace("-", "_"), label=label_of(setting),
        n_players=n_players, scores=cos_ema.tolist(), gt=gt.tolist(),
        contaminated=[bool(x) for x in contaminated], pcc=history[-1],
        pcc_history=history, rounds=rounds, accuracy=float(accuracy),
        elapsed=time.perf_counter() - started, source="real",
        timing=timer.to_dict(rounds=rounds, seeds=1, source="measured"),
        gt_seconds=gt_seconds(key),
    )


# --------------------------------------------------------------------------
# Result cache
# --------------------------------------------------------------------------


def cache_path(method: str, key: str, rounds: int | None) -> Path:
    return config.CACHE_DIR / f"{method}__{key}__r{rounds or 0}.json"


def load_cached(method: str, key: str, rounds: int | None):
    path = cache_path(method, key, rounds)
    if not path.is_file():
        return None
    result = Result.from_dict(json.loads(path.read_text()))
    result.source = "cache"
    if result.timing is None:  # cache written before timing was recorded
        result.timing = _timing_from_elapsed(result)
    if result.gt_seconds is None:
        result.gt_seconds = gt_seconds(key)
    return result


def save_cached(result: Result) -> Path:
    path = cache_path(result.method, result.key, result.rounds)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(result.to_dict(), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


# --------------------------------------------------------------------------
# Job driver (worker thread)
# --------------------------------------------------------------------------


def run_job(method: str, mode: str, dataset: str, label: str, rounds: int, device: str,
            events, cancel=None, *, round_delay: float | None = None,
            stage_delay: float | None = None) -> None:
    """Evaluate `method` on one setting (`dataset`, `label`) and put events on
    `events`. `mode` is "real" or "replay"; `rounds` applies to real TRAC runs
    only. Ends with exactly one of `done`, `cancelled` or `error`.
    """

    put = events.put
    try:
        setting = setting_for(dataset, label)
        key = setting["key"]
        title = config.DATASET_TITLES[dataset]
        is_trac = method == "trac"
        total = rounds if (is_trac and mode == "real") else (
            min(int(stored_trac_row(key)["rounds"]), config.FULL_TRAC_ROUNDS[dataset])
            if is_trac else len(texts.PARZEN_STAGES)
        )
        put({"type": "dataset_start", "dataset": dataset, "label": label, "max": total})
        put({"type": "log", "dataset": dataset,
             "text": texts.LOG_DATASET_START.format(title=title, label=label)})

        def progress(event):
            put({**event, "dataset": dataset})

        _check(cancel)
        if mode == "replay":
            result = (
                replay_trac(key, progress, cancel, round_delay) if is_trac
                else replay_parzen(key, progress, cancel, stage_delay)
            )
        else:
            result = load_cached(method, key, rounds if is_trac else None)
            if result is not None:
                put({"type": "log", "dataset": dataset, "text": texts.LOG_CACHE_HIT})
                put({"type": "progress", "dataset": dataset, "value": total, "max": total,
                     "text": ""})
            else:
                result = (
                    compute_trac(key, rounds, device, progress, cancel) if is_trac
                    else compute_parzen(key, device, progress, cancel)
                )
                save_cached(result)
        put({"type": "log", "dataset": dataset,
             "text": texts.LOG_DATASET_DONE.format(
                 title=title, label=label, pcc=result.pcc, elapsed=result.elapsed)})
        put({"type": "result", "dataset": dataset, "result": result})
        put({"type": "done"})
    except Cancelled:
        put({"type": "cancelled"})
    except Exception:  # noqa: BLE001 - surfaced to the UI as a message box
        put({"type": "error", "text": traceback.format_exc()})
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
