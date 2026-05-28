# Copyright (c) 2014-2026, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LBANN/ScaFFold.
#
# SPDX-License-Identifier: (Apache-2.0)

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from mpi4py import MPI

from ScaFFold.datagen import volumegen

META_FILENAME = "meta.yaml"
DATASET_FORMAT_VERSION = 3
V2_DATASET_FORMAT_VERSION = 2
V2_INCLUDE_KEYS = [
    "dataset_format_version",
    "n_categories",
    "n_instances_used_per_fractal",
    "problem_scale",
    "seed",
    "variance_threshold",
    "n_fracts_per_vol",
    "val_split",
]
INCLUDE_KEYS = V2_INCLUDE_KEYS + [
    "dc_num_shards",
    "dc_shard_dims",
]


def canonicalize(input):
    """
    Sort dict keys, recursing on lists/dicts, for stable hashing.
    """
    if isinstance(input, dict):
        return {key: canonicalize(input[key]) for key in sorted(input)}
    elif isinstance(input, list):
        return [canonicalize(item) for item in input]
    else:
        return input


def _get_required_keys_dict(
    config: Dict[str, Any], include_keys: list[str]
) -> Dict[str, Any]:
    """
    Build a dict containing only the required keys.
    Raises KeyError if any required key is missing.
    """
    missing = [key for key in include_keys if key not in config]
    if missing:
        raise KeyError(
            f"Missing expected top-level keys in run YAML: {missing}. "
            f"Required INCLUDE_KEYS={include_keys}"
        )
    required = {key: config[key] for key in include_keys}
    return canonicalize(required)


def _canonicalize_v3_shard_layout(volume_config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize shard layout ordering so equivalent v3 layouts share cache IDs."""

    canonical_config = volume_config.copy()
    num_shards = canonical_config["dc_num_shards"]
    shard_dims = canonical_config["dc_shard_dims"]
    if len(num_shards) != len(shard_dims):
        raise ValueError(
            f"dc_num_shards {num_shards} must have same length as dc_shard_dims {shard_dims}"
        )

    shard_layout = sorted(
        (int(shard_dim), int(num_shard))
        for num_shard, shard_dim in zip(num_shards, shard_dims)
    )
    canonical_config["dc_shard_dims"] = [shard_dim for shard_dim, _ in shard_layout]
    canonical_config["dc_num_shards"] = [num_shard for _, num_shard in shard_layout]
    return canonical_config


def _hash_volume_config(volume_config: Dict[str, Any]) -> str:
    s = json.dumps(volume_config, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(s).hexdigest()[:12]


def _volume_config_for_version(
    config_dict, dataset_format_version, canonicalize_v3_shard_layout=True
):
    versioned_config = config_dict.copy()
    versioned_config["dataset_format_version"] = dataset_format_version
    if dataset_format_version == DATASET_FORMAT_VERSION:
        include_keys = INCLUDE_KEYS
    else:
        include_keys = V2_INCLUDE_KEYS
    volume_config = _get_required_keys_dict(
        config=versioned_config,
        include_keys=include_keys,
    )
    if (
        dataset_format_version == DATASET_FORMAT_VERSION
        and canonicalize_v3_shard_layout
    ):
        volume_config = _canonicalize_v3_shard_layout(volume_config)
    return volume_config


def _requested_unsharded_layout(config_dict: Dict[str, Any]) -> bool:
    total_shards = 1
    for value in config_dict["dc_num_shards"]:
        total_shards *= int(value)
    return total_shards == 1


def _git_commit_short() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,  # Don't show console output to user
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        print(
            "Tried to get git commit id in non-git repo. No commit id will be enforced for dataset reuse."
        )
        return "no-commit-id"
    except Exception:
        print(
            "Exception when trying to get git commit for dataset. No commit id will be enforced for dataset reuse."
        )
        return "no-commit-id"


def _find_reusable_dataset(
    root: Path,
    config_id: str,
    dataset_format_version: int,
    commit: str,
    require_commit: bool,
) -> Optional[Path]:
    base = root / config_id
    if not base.exists():
        return None

    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True
    )
    for dataset_path in candidates:
        meta_path = dataset_path / META_FILENAME
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text()) or {}
        if meta.get("config_id") != config_id:
            continue
        if meta.get("dataset_format_version", 1) != dataset_format_version:
            continue
        if require_commit and meta.get("code_commit") != commit:
            continue
        return dataset_path

    return None


def get_dataset(
    config: Namespace,
    require_commit: bool = False,  # default: ignore commit mismatches for reuse
) -> Path:
    """
    Get dataset matching requested config, either by:
        1. Finding an existing dataset with matching config
            (optionally enforcing matching code commits), or
        2. Generating a new dataset from the input config.
    Allows for reusing existing datasets where appropriate.

    Returns: Path to the selected (or newly created) dataset directory.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    root = Path(config.dataset_dir)
    root.mkdir(exist_ok=True)

    # V3 is the current physical-shard format. The physical dataset layout is
    # defined by dc_num_shards/dc_shard_dims, matching the DistConv layout.
    config_dict = vars(config).copy()
    volume_config = _volume_config_for_version(config_dict, DATASET_FORMAT_VERSION)
    metadata_volume_config = _volume_config_for_version(
        config_dict,
        DATASET_FORMAT_VERSION,
        canonicalize_v3_shard_layout=False,
    )
    config_id = _hash_volume_config(volume_config)
    v2_volume_config = _volume_config_for_version(config_dict, V2_DATASET_FORMAT_VERSION)
    v2_config_id = _hash_volume_config(v2_volume_config)
    commit = _git_commit_short()

    # Prefer a matching V3 physical-shard dataset.
    dataset_path = _find_reusable_dataset(
        root,
        config_id,
        DATASET_FORMAT_VERSION,
        commit,
        require_commit,
    )
    if dataset_path is not None:
        print(
            "Valid existing v3 sharded dataset found. Reusing this dataset..."
        )  # FIXME replace with updated logging
        return dataset_path

    # V2 datasets are full-volume files without shard suffixes. Reuse them only
    # for unsharded requests so sharded generation never silently returns a
    # cache that lacks the requested shard files.
    if _requested_unsharded_layout(config_dict):
        dataset_path = _find_reusable_dataset(
            root,
            v2_config_id,
            V2_DATASET_FORMAT_VERSION,
            commit,
            require_commit,
        )
        if dataset_path is not None:
            print(
                "Valid existing v2 full-volume dataset found. Reusing this dataset..."
            )  # FIXME replace with updated logging
            return dataset_path

    # Otherwise, generate a new dataset
    base = root / config_id
    print(f"No valid existing dataset found at {base}. Generating new dataset...")
    if rank == 0:
        base.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest = base / f"{ts}__{commit}"
        tmp = base / f".tmp_{ts}"
        tmp.mkdir(parents=True, exist_ok=False)
    else:
        tmp = dest = None

    # broadcast the staging + final paths
    tmp, dest = comm.bcast((tmp, dest) if rank == 0 else None, root=0)

    config.dataset_dir = tmp
    ok = True
    err = ""

    try:
        volumegen.main(config)
    except Exception as e:
        ok = False
        err = f"volumegen attempt failed: rank {rank}: {type(e).__name__}: {e}"

    # Check that all ranks succeeded in volumegen, then sync
    all_ok = comm.allreduce(1 if ok else 0, op=MPI.MIN) == 1
    errs = comm.gather(err, root=0)

    failure_msg = None
    if rank == 0:
        if not all_ok:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
            msgs = "; ".join(e for e in errs if e)
            failure_msg = f"dataset generation failed: {msgs or 'unknown error'}"

    failure_msg = comm.bcast(failure_msg, root=0)
    if failure_msg:
        raise RuntimeError(failure_msg)

    # rank 0 has file write + move
    finalize_err = ""
    if rank == 0:
        try:
            # Write to tmp, then move, so readers never see half-written dataset
            meta = {
                "config_id": config_id,
                "dataset_format_version": DATASET_FORMAT_VERSION,
                "config_subset": metadata_volume_config,
                "include_keys": INCLUDE_KEYS,
                "code_commit": commit,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (tmp / META_FILENAME).write_text(
                yaml.safe_dump(meta, sort_keys=True, default_flow_style=False)
            )
            tmp.rename(dest)
        except Exception as e:
            finalize_err = (
                f"dataset finalization failed: rank 0: {type(e).__name__}: {e}"
            )

    finalize_err = comm.bcast(finalize_err, root=0)
    if finalize_err:
        if rank == 0:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
        raise RuntimeError(finalize_err)

    # ensure the rename is visible everywhere before returning
    comm.Barrier()
    return dest
