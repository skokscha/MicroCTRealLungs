"""Prototype: through-paths and bottleneck search in the air phase (510^3).

Percolation via wavefront search (Dijkstra = MCP from skimage, C backend):
- strictly 6-connectivity (face neighbours only, no diagonals) - MCP by design
- costs: air voxel = 1 step, tissue = 1e9 (never traversed when a path exists)
- multi-source wave from ALL air voxels of the inlet face -> distance field
- shortest through-path length and tortuosity for each of the 3 axes

Bottleneck ("gorge") tracking - the idea from the meeting: cells remember how
many trees passed through them. We grow K shortest paths from random outlet
voxels back to the inlet and count visits per voxel. Voxels crossed by many
independent paths = narrow places all routes are forced through.

Outputs (data/preview/):
  connectivity_paths.npz  - visit counts (int32 510^3) + stats per axis
  prints stats for the report
"""
import time
from pathlib import Path

import numpy as np
from skimage.graph import MCP

PROJECT = Path(__file__).resolve().parent.parent
PREVIEW = PROJECT / "data" / "preview"
TISSUE_COST = 1e9
K_PATHS = 128  # paths per axis for the visit-counter


def axis_run(air: np.ndarray, axis: int, rng) -> dict:
    """Wave from face 0 to face -1 along `axis`; returns stats + counter."""
    t0 = time.time()
    costs = np.where(air, 1.0, TISSUE_COST).astype(np.float64)
    # inlet face in the moved frame -> remap coordinates back to the original
    face_in = np.moveaxis(air, axis, 0)[0]
    others = [i for i in range(3) if i != axis]
    starts = []
    for c in np.argwhere(face_in):
        s = [0, 0, 0]
        s[axis] = 0
        s[others[0]] = int(c[0])
        s[others[1]] = int(c[1])
        starts.append(tuple(s))
    mcp = MCP(costs)
    dist, _ = mcp.find_costs(starts)
    del costs
    face_out = np.moveaxis(air, axis, 0)[-1]
    d_out = np.moveaxis(dist, axis, 0)[-1]
    reach = face_out & np.isfinite(d_out) & (d_out < TISSUE_COST / 2)
    L = float(d_out[reach].min()) if reach.any() else float("nan")

    stats = {
        "axis": axis,
        "n_start": len(starts),
        "n_out_air": int(face_out.sum()),
        "n_out_reached": int(reach.sum()),
        "shortest_path_vox": L,
        "tortuosity": L / 509.0 if reach.any() else float("nan"),
    }
    print(f"axis {axis}: starts={len(starts):,}, outlet air={face_out.sum():,}, "
          f"reached={reach.sum():,} ({100 * reach.sum() / max(face_out.sum(), 1):.1f}%), "
          f"shortest path={L:.0f} vox, tortuosity={stats['tortuosity']:.2f}, "
          f"{time.time() - t0:.0f}s", flush=True)

    counter = np.zeros(air.shape, dtype=np.int32)
    tortuosities = []
    example_path = None
    # trace K random reached outlet voxels back to the inlet
    if reach.any():
        out_coords = np.argwhere(reach)  # 2D coords on the outlet face (moved frame)
        others = [i for i in range(3) if i != axis]
        pick = rng.choice(len(out_coords), min(K_PATHS, len(out_coords)), replace=False)
        for e in out_coords[pick]:
            end = [0, 0, 0]  # outlet face has index 509 on the flow axis
            end[axis] = 509
            end[others[0]] = int(e[0])
            end[others[1]] = int(e[1])
            path = mcp.traceback(tuple(end))
            idx = tuple(np.array(path).T)
            counter[idx] += 1
            tortuosities.append(len(path) / 509.0)
            if example_path is None:
                example_path = np.array(path, dtype=np.int16)
        stats["mean_tortuosity"] = float(np.mean(tortuosities))
        stats["n_path_voxels"] = int((counter > 0).sum())
        stats["max_visits"] = int(counter.max())
        stats["gorge_voxels"] = int((counter >= 0.5 * len(tortuosities)).sum())
        print(f"  {len(tortuosities)} paths traced, mean tortuosity "
              f"{stats['mean_tortuosity']:.2f}, visited voxels {stats['n_path_voxels']:,}, "
              f"max visits {stats['max_visits']}, voxels crossed by >=50% paths: "
              f"{stats['gorge_voxels']:,}", flush=True)
    del dist
    return stats, counter, example_path


def main() -> None:
    air = np.load(PREVIEW / "air_phase_voxels_510.npz")["air"]
    rng = np.random.default_rng(0)

    all_stats = {}
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        stats, counter, example_path = axis_run(air, axis, rng)
        all_stats[name] = stats
        np.savez_compressed(
            PREVIEW / f"connectivity_visits_{name}.npz",
            visits=counter,
            example_path=example_path if example_path is not None else np.zeros((0, 3), np.int16),
            **{k: np.float64(v) for k, v in stats.items() if isinstance(v, float)},
        )
        del counter

    print("\n=== summary ===")
    for name, s in all_stats.items():
        print(name, s, flush=True)


if __name__ == "__main__":
    main()
