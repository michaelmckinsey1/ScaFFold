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

import logging
import math
import os
import pickle
import random
import time
from math import ceil
from typing import Callable, Dict

import numpy as np
from mpi4py import MPI

from ScaFFold.utils.config_utils import Config
from ScaFFold.utils.data_types import DEFAULT_NP_DTYPE, MASK_DTYPE, VOLUME_DTYPE
from ScaFFold.utils.spatial_sharding import (
    normalize_sharding,
    shard_file_suffix,
    shard_id_to_indices,
    spatial_slices,
    total_shards,
)


def load_np_ptcloud(path: str) -> np.ndarray:
    """
    Read a .npy file and return an (N,3) array of dtype float64.
    """
    pts = np.load(path)
    return pts.astype(DEFAULT_NP_DTYPE, copy=False)


def points_to_voxelgrid(
    points: np.ndarray, grid_size: int, eps: float = 1e-6
) -> np.ndarray:
    """
    Convert an (N,3) float64 point cloud directly into a boolean voxel grid
    of shape (grid_size, grid_size, grid_size).
    """
    idx = points_to_voxel_indices(points, grid_size, eps=eps)

    # Scatter into a boolean grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    return grid


def points_to_voxel_indices(
    points: np.ndarray, grid_size: int, eps: float = 1e-6
) -> np.ndarray:
    """
    Convert an (N,3) point cloud into global voxel indices using the same
    math as points_to_voxelgrid(), without allocating a full boolean grid.
    """

    # 1) Axis-aligned bounding box in float64
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    # 2) Voxel size per dimension (float64)
    voxel_size = (maxs - mins + eps) / grid_size

    # 3) Map points into [0,grid_size) indices
    scaled = (points - mins) / voxel_size
    np.floor(scaled, out=scaled)
    idx = scaled.astype(int)

    # 4) Clip to valid range
    np.clip(idx, 0, grid_size - 1, out=idx)

    return idx


def _point_cloud_path(config, curr_category: int, curr_instance: int) -> str:
    """Return the input point-cloud path for a fractal instance."""

    instances_dir = f"var{config.variance_threshold}/instances/np{config.point_num}"
    return os.path.join(
        str(config.fract_base_dir),
        instances_dir,
        f"{curr_category:06d}",
        f"{curr_category:06d}_{curr_instance:04d}.npy",
    )


def _local_shape(slices):
    """Return the local spatial shape described by shard slices."""

    return tuple(s.stop - s.start for s in slices)


def _physical_sharding(config):
    """Return normalized physical sharding from the generation config."""

    return normalize_sharding(config.dc_num_shards, config.dc_shard_dims)


def _validate_generation_config(config):
    """Validate sharded generation settings and return normalized layout data."""

    num_shards, shard_dims = _physical_sharding(config)
    n_total_shards = total_shards(num_shards)

    grid_size = math.floor(config.vol_size * config.scale)
    if grid_size != config.vol_size:
        raise ValueError(
            "Sharded volume generation currently requires config.scale == 1 so shard files tile the full volume"
        )

    return num_shards, shard_dims, n_total_shards, grid_size


def _voxelized_fractals_for_volume(
    config,
    curr_vol: np.ndarray,
    fractal_colors: np.ndarray,
    point_cloud_loader: Callable[[str], np.ndarray] = load_np_ptcloud,
):
    """Load and voxelize all fractals needed for one logical volume."""

    n_fracts_per_vol = config.n_fracts_per_vol
    grid_size = math.floor(config.vol_size * config.scale)
    voxelized_fractals = []

    for curr_fract in range(n_fracts_per_vol):
        curr_category = int(curr_vol[1 + 2 * curr_fract])
        curr_instance = int(curr_vol[1 + 2 * curr_fract + 1])
        fractal_color = fractal_colors[curr_category]

        point_cloud_path = _point_cloud_path(config, curr_category, curr_instance)
        if point_cloud_loader is load_np_ptcloud and not os.path.exists(
            point_cloud_path
        ):
            raise FileNotFoundError(
                f"File {point_cloud_path} does not exist. Ensure you have run 'scaffold generate_fractals ...'"
            )

        points = point_cloud_loader(point_cloud_path)
        idx = points_to_voxel_indices(points, grid_size)
        voxelized_fractals.append((curr_category, fractal_color, idx))

    return voxelized_fractals


def _render_volume_shard(config, voxelized_fractals, shard_id: int):
    """Render one physical shard from precomputed global voxel indices."""

    num_shards, shard_dims = _physical_sharding(config)
    shard_indices = shard_id_to_indices(shard_id, num_shards)
    slices = spatial_slices(
        (config.vol_size, config.vol_size, config.vol_size),
        shard_dims,
        num_shards,
        shard_indices,
    )
    local_shape = _local_shape(slices)

    volume = np.full((3, *local_shape), 0, dtype=VOLUME_DTYPE)
    mask = np.full(local_shape, 0, dtype=MASK_DTYPE)

    for curr_category, fractal_color, idx in voxelized_fractals:
        keep = np.ones(idx.shape[0], dtype=bool)
        for axis, axis_slice in enumerate(slices):
            keep &= idx[:, axis] >= axis_slice.start
            keep &= idx[:, axis] < axis_slice.stop

        if not np.any(keep):
            continue

        local_idx = idx[keep]
        local_idx[:, 0] -= slices[0].start
        local_idx[:, 1] -= slices[1].start
        local_idx[:, 2] -= slices[2].start
        d = local_idx[:, 0]
        h = local_idx[:, 1]
        w = local_idx[:, 2]

        volume[:, d, h, w] = fractal_color[:, None]
        mask[d, h, w] = curr_category + 1

    return volume, mask


def main(config: Dict):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dataset_dir = str(config.dataset_dir)

    vol_path = os.path.join(dataset_dir, "volumes")
    mask_path = os.path.join(dataset_dir, "masks")
    volumes_contents_path = os.path.join(dataset_dir, "volumes_contents.csv")

    n_fracts_per_vol = config.n_fracts_per_vol
    _, _, n_total_shards, _ = _validate_generation_config(config)

    random.seed(config.seed)  # Python
    np.random.seed(config.seed)  # NumPy

    # Set up directories and select instances from each category
    volumes_contents = None
    setup_err = ""

    if rank == 0:
        try:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            for subdir in ["training", "validation"]:
                os.makedirs(os.path.join(vol_path, subdir), exist_ok=True)
                os.makedirs(os.path.join(mask_path, subdir), exist_ok=True)

            # Force n_instances_used_per_fractal to be multiple of n_fracts_per_vol
            if config.n_instances_used_per_fractal % n_fracts_per_vol != 0:
                print(
                    f"volumegen.py: WARNING: n_instances_used_per_fractal ({config.n_instances_used_per_fractal}) \n"
                    f"NOT multiple of n_fracts_per_vol={n_fracts_per_vol}. Rounding down."
                )
                config.n_instances_used_per_fractal = (
                    config.n_instances_used_per_fractal
                    // n_fracts_per_vol
                    * n_fracts_per_vol
                )

            # Randomly select n_instances_used_per_fractal instances from each fractal class.
            instances_list = []
            for category in range(config.n_categories):
                instances_remaining = config.n_instances_used_per_fractal
                random_instances = []
                while instances_remaining > 0:
                    random_instances.extend(
                        random.sample(range(145), min(145, instances_remaining))
                    )
                    instances_remaining -= min(145, instances_remaining)

                category_instance_pairs = [
                    [category, instance] for instance in random_instances
                ]
                instances_list.extend(category_instance_pairs)

            instances_list = np.array(instances_list, dtype=int)
            np.random.shuffle(instances_list)

            volumes_contents = instances_list.reshape(-1, 2 * n_fracts_per_vol)

            indices = np.arange(volumes_contents.shape[0]).reshape(-1, 1)
            volumes_contents = np.hstack([indices, volumes_contents])

            with open(volumes_contents_path, "wb") as f:
                np.savetxt(f, volumes_contents.astype(int), fmt="%i", delimiter=",")
            print(
                f"volumegen.py({rank}): finished writing volumes_contents (shape = {volumes_contents.shape})"
            )
        except Exception as e:
            setup_err = f"setup failed: rank {rank}: {type(e).__name__}: {e}"

    # Broadcast to all ranks
    volumes_contents, setup_err = comm.bcast((volumes_contents, setup_err), root=0)
    if setup_err:
        raise RuntimeError(setup_err)

    # Rank 0 creates shared metadata above; wait before local writer setup.
    comm.Barrier()

    for subdir in ["training", "validation"]:
        os.makedirs(os.path.join(vol_path, subdir), exist_ok=True)
        os.makedirs(os.path.join(mask_path, subdir), exist_ok=True)

    # Wait until every rank has ensured the writer directories exist.
    comm.Barrier()

    # Determine train/val split globally so all ranks know where to save
    num_volumes = len(volumes_contents)
    random.seed(config.seed)
    val_indices = set(
        random.sample(range(num_volumes), int(num_volumes * config.val_split / 100))
    )

    # Work distribution: each task renders one physical shard of one logical volume.
    total_tasks = num_volumes * n_total_shards
    stride = ceil(total_tasks / size)
    start_idx = rank * stride
    end_idx = min(((rank + 1) * stride), total_tasks)

    generation_err = ""
    n_generated_shards = 0
    try:
        if start_idx >= end_idx:
            logging.info(f"Rank {rank} given no physical shard tasks to generate")

        else:
            print(
                f"rank {rank} responsible for physical shard tasks "
                f"{start_idx} through {end_idx - 1}"
            )

            np.random.seed(config.seed)
            fractal_colors = np.random.rand(
                max(config.n_categories, n_fracts_per_vol), 3
            )

            # Generation loop
            start_time = time.time()
            cached_volume_idx = None
            cached_global_vol_idx = None
            cached_voxelized_fractals = None

            for i, task_idx in enumerate(range(start_idx, end_idx)):
                if i % 10 == 0:
                    logging.info(
                        f"Rank {rank} processing local physical shard task {i}..."
                    )

                volume_idx = task_idx // n_total_shards
                shard_id = task_idx % n_total_shards

                if cached_volume_idx != volume_idx:
                    curr_vol = volumes_contents[volume_idx]
                    global_vol_idx = int(curr_vol[0])
                    vol_seed = config.seed + global_vol_idx
                    random.seed(vol_seed)
                    np.random.seed(vol_seed)

                    cached_voxelized_fractals = _voxelized_fractals_for_volume(
                        config,
                        curr_vol,
                        fractal_colors,
                    )
                    cached_volume_idx = volume_idx
                    cached_global_vol_idx = global_vol_idx

                volume_to_save, mask_to_save = _render_volume_shard(
                    config,
                    cached_voxelized_fractals,
                    shard_id,
                )

                # Determine destination folder
                subdir = (
                    "validation"
                    if cached_global_vol_idx in val_indices
                    else "training"
                )
                shard_suffix = shard_file_suffix(shard_id)

                vol_file = os.path.join(
                    vol_path, subdir, f"{cached_global_vol_idx}{shard_suffix}.npy"
                )
                with open(vol_file, "wb") as f:
                    np.save(f, volume_to_save)

                mask_file = os.path.join(
                    mask_path,
                    subdir,
                    f"{cached_global_vol_idx}{shard_suffix}_mask.npy",
                )
                with open(mask_file, "wb") as f:
                    np.save(f, mask_to_save)
                n_generated_shards += 1

            end_time = time.time()
            total_time = end_time - start_time
            if rank == 0:
                shard_rate = n_generated_shards / total_time
                print(
                    f"Rank 0 generated {n_generated_shards} volume shards "
                    f"from {end_idx - start_idx} physical shard tasks in "
                    f"{total_time:.2f} seconds | {shard_rate:.2f} shards per second"
                )
    except Exception as e:
        generation_err = (
            f"volume shard generation failed: rank {rank}: {type(e).__name__}: {e}"
        )

    all_generated = comm.allreduce(1 if not generation_err else 0, op=MPI.MIN) == 1
    generation_errs = comm.gather(generation_err, root=0)
    generation_failure = ""
    if rank == 0 and not all_generated:
        msgs = "; ".join(e for e in generation_errs if e)
        generation_failure = msgs or "unknown volume shard generation error"
    generation_failure = comm.bcast(generation_failure, root=0)
    if generation_failure:
        raise RuntimeError(generation_failure)

    # Barrier to ensure all ranks are finished writing
    comm.Barrier()

    if rank == 0:
        print(f"volumegen.py({rank}): All ranks done. Proceeding to split.")

    # Do the train/val split and generate lists of unique train/val masks
    if rank == 0:
        print("volumegen.py: volume gen COMPLETE. Now generating unique mask lists")

        # Directories are already created at start of script

        # Reconstruct lists for unique mask value scanning
        val_files = sorted(list(val_indices))
        train_files = sorted(list(set(range(num_volumes)) - val_indices))

        print(
            f"volumegen.py({rank}): len(val_files)={len(val_files)}, len(train_files)={len(train_files)}."
        )

        # Save lists of unique train and val mask values
        print(
            f"volumegen.py({rank}): calculating unique mask values from configuration (no file read)"
        )

        # volumes_contents layout is [vol_idx, cat1, inst1, cat2, inst2, ...]
        # We want the categories, which are at indices 1, 3, 5, etc.
        cat_cols = slice(1, None, 2)

        # Process unique train mask values
        # Extract rows corresponding to train files
        train_rows = volumes_contents[train_files]
        # Extract only the category columns and flatten to a 1D array
        train_cats = train_rows[:, cat_cols].flatten()
        # Create set of unique labels: (category + 1) and background (0)
        unique_train = set(train_cats + 1)
        unique_train.add(0)

        unique_train = sorted(list(unique_train))
        unique_train_file = f"{dataset_dir}/train_unique_mask_vals"
        with open(unique_train_file, "wb") as out:
            pickle.dump({"mask_values": unique_train}, out)

        # Process unique val mask values (same logic)
        val_rows = volumes_contents[val_files]
        val_cats = val_rows[:, cat_cols].flatten()
        unique_val = set(val_cats + 1)
        unique_val.add(0)

        unique_val = sorted(list(unique_val))
        unique_val_file = f"{dataset_dir}/val_unique_mask_vals"
        with open(unique_val_file, "wb") as out:
            pickle.dump({"mask_values": unique_val}, out)


if __name__ == "__main__":
    import yaml

    with open("run_config.yaml") as f:
        config_dict = yaml.full_load(f)
    config = Config(config_dict)
    main(config)
