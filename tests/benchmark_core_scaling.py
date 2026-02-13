"""
Core-scaling benchmark for ESFS CPU optimizations.

Tests performance at core counts: 2, 4, 6, 8, 10, 12, 14, 16.
Compares the fused overlap+ESS kernel against the old separate path.

Run with:  python tests/benchmark_core_scaling.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import scipy.sparse as spsparse
import time
from numba import set_num_threads, get_num_threads


def make_benchmark_data(n_samples=5000, n_fixed=100, n_features=1000, density=0.15, seed=42):
    """Create benchmark-sized sparse test data."""
    rng = np.random.default_rng(seed)

    ff_dense = rng.random((n_samples, n_fixed)) * (rng.random((n_samples, n_fixed)) < density)
    gs_dense = rng.random((n_samples, n_features)) * (rng.random((n_samples, n_features)) < density)

    fixed_features = spsparse.csc_matrix(ff_dense)
    fixed_features.sort_indices()
    global_scaled = spsparse.csc_matrix(gs_dense)
    global_scaled.sort_indices()

    sample_cardinality = n_samples
    ff_cardinality = np.asarray(fixed_features.sum(axis=0)).flatten()
    gs_sums = np.asarray(global_scaled.sum(axis=0)).flatten()

    ff_minority = ff_cardinality.copy()
    ff_minority[ff_minority >= sample_cardinality / 2] = sample_cardinality - ff_minority[ff_minority >= sample_cardinality / 2]
    gs_minority = gs_sums.copy()
    gs_minority[gs_minority >= sample_cardinality / 2] = sample_cardinality - gs_minority[gs_minority >= sample_cardinality / 2]

    FF_QF_vs_RF = ff_minority[:, None] > gs_minority[None, :]
    RFms = np.where(~FF_QF_vs_RF, ff_minority[:, None], gs_minority[None, :])
    QFms = np.where(FF_QF_vs_RF, ff_minority[:, None], gs_minority[None, :])
    RFMs = sample_cardinality - RFms
    QFMs = sample_cardinality - QFms

    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = np.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])

    return (fixed_features, global_scaled, sample_cardinality,
            RFms, QFms, RFMs, QFMs, max_ent_options,
            ff_cardinality, gs_sums, FF_QF_vs_RF)


def run_old_full(fixed_features, global_scaled, sample_cardinality,
                 RFms, QFms, RFMs, QFMs, max_ent_options,
                 ff_cardinality, gs_sums, FF_QF_vs_RF):
    """Run old path: separate overlap + vectorized ESS."""
    from esfs.ESFS import (overlaps_and_inverse_sparse, calc_ESSs_vec, nanmaximum,
                           _build_case_tables)

    # Overlap computation
    ff_csc = fixed_features.tocsc()
    gs_csc = global_scaled.tocsc()
    ff_data = np.ascontiguousarray(ff_csc.data, dtype=np.float32)
    ff_indices = np.ascontiguousarray(ff_csc.indices, dtype=np.int32)
    ff_indptr = np.ascontiguousarray(ff_csc.indptr, dtype=np.int32)
    gs_data = np.ascontiguousarray(gs_csc.data, dtype=np.float32)
    gs_indices = np.ascontiguousarray(gs_csc.indices, dtype=np.int32)
    gs_indptr = np.ascontiguousarray(gs_csc.indptr, dtype=np.int32)

    overlaps, inverse_overlaps = overlaps_and_inverse_sparse(
        ff_data, ff_indices, ff_indptr,
        gs_data, gs_indices, gs_indptr,
        fixed_features.shape[0], fixed_features.shape[1], global_scaled.shape[1],
        use_float64=False
    )

    # Case tables
    case_idxs, case_patterns, overlap_lookup = _build_case_tables(
        ff_cardinality, sample_cardinality, gs_sums, FF_QF_vs_RF)

    # ESS computation
    ESSs, D_EPs, O_EPs, SWs, SGs = calc_ESSs_vec(
        RFms, QFms, RFMs, QFMs, max_ent_options,
        sample_cardinality, overlaps, inverse_overlaps,
        case_idxs, case_patterns, overlap_lookup,
        xp_mod=np,
    )
    iden_feats, iden_cols = np.nonzero(ESSs == 1)
    D_EPs[iden_feats, iden_cols] = 0
    O_EPs[iden_feats, iden_cols] = 0
    EPs = nanmaximum(D_EPs, O_EPs)
    return ESSs, EPs, SWs, SGs


def run_fused(fixed_features, global_scaled, sample_cardinality,
              RFms, QFms, RFMs, QFMs, max_ent_options,
              ff_cardinality, gs_sums, FF_QF_vs_RF):
    """Run fused overlap+ESS kernel."""
    from esfs.ESFS import _fused_overlap_ess_numba, _build_case_tables

    case_idxs, case_patterns, overlap_lookup = _build_case_tables(
        ff_cardinality, sample_cardinality, gs_sums, FF_QF_vs_RF)

    ff_csc = fixed_features.tocsc()
    gs_csc = global_scaled.tocsc()
    ff_data = np.ascontiguousarray(ff_csc.data, dtype=np.float32)
    ff_indices = np.ascontiguousarray(ff_csc.indices, dtype=np.int32)
    ff_indptr = np.ascontiguousarray(ff_csc.indptr, dtype=np.int32)
    gs_data = np.ascontiguousarray(gs_csc.data, dtype=np.float32)
    gs_indices = np.ascontiguousarray(gs_csc.indices, dtype=np.int32)
    gs_indptr = np.ascontiguousarray(gs_csc.indptr, dtype=np.int32)

    ESSs, EPs, SWs, SGs = _fused_overlap_ess_numba(
        ff_data, ff_indices, ff_indptr,
        gs_data, gs_indices, gs_indptr,
        fixed_features.shape[1], global_scaled.shape[1],
        RFms, QFms, RFMs, QFMs, max_ent_options,
        sample_cardinality,
        case_idxs, case_patterns, overlap_lookup,
    )
    return ESSs, EPs, SWs, SGs


def main():
    core_counts = [2, 4, 6, 8, 10, 12, 14, 16]
    max_threads = get_num_threads()
    core_counts = [c for c in core_counts if c <= max_threads]

    print("=" * 70)
    print(f"ESFS Core Scaling Benchmark (max {max_threads} threads available)")
    print("=" * 70)

    # Generate data
    print("\nGenerating benchmark data...")
    data = make_benchmark_data(n_samples=5000, n_fixed=100, n_features=1000, density=0.15)
    (ff, gs, sc, RFms, QFms, RFMs, QFMs, meo, ffc, gss, ffqr) = data
    print(f"  Fixed features: {ff.shape}, Global scaled: {gs.shape}")
    print(f"  Work items: {ff.shape[1]} x {gs.shape[1]} = {ff.shape[1] * gs.shape[1]:,}")

    # JIT warmup
    print("\nWarming up JIT...")
    set_num_threads(min(2, max_threads))
    run_old_full(ff[:10, :2].tocsc(), gs[:10, :2].tocsc(), sc,
                 RFms[:2, :2], QFms[:2, :2], RFMs[:2, :2], QFMs[:2, :2],
                 meo[:, :2, :2], ffc[:2], gss[:2], ffqr[:2, :2])
    small_ff = spsparse.csc_matrix(ff[:10, :2].toarray())
    small_gs = spsparse.csc_matrix(gs[:10, :2].toarray())
    small_ff.sort_indices()
    small_gs.sort_indices()
    run_fused(small_ff, small_gs, sc,
              RFms[:2, :2], QFms[:2, :2], RFMs[:2, :2], QFMs[:2, :2],
              meo[:, :2, :2], ffc[:2], gss[:2], ffqr[:2, :2])
    print("  Warmup complete.")

    # Benchmark
    n_repeats = 3
    old_times = {}
    fused_times = {}

    print(f"\nBenchmarking ({n_repeats} repeats each)...\n")
    print(f"{'Cores':>6} | {'Old (s)':>10} | {'Fused (s)':>10} | {'Speedup':>8} | {'Scaling':>8}")
    print("-" * 56)

    for nc in core_counts:
        set_num_threads(nc)

        # Old path
        times_old = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            run_old_full(ff, gs, sc, RFms, QFms, RFMs, QFMs, meo, ffc, gss, ffqr)
            times_old.append(time.perf_counter() - t0)
        t_old = min(times_old)
        old_times[nc] = t_old

        # Fused path
        times_fused = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            run_fused(ff, gs, sc, RFms, QFms, RFMs, QFMs, meo, ffc, gss, ffqr)
            times_fused.append(time.perf_counter() - t0)
        t_fused = min(times_fused)
        fused_times[nc] = t_fused

        speedup = t_old / t_fused
        # Parallel scaling efficiency relative to 2-core fused time
        if nc == core_counts[0]:
            base_fused = t_fused * core_counts[0]
        scaling = (base_fused / t_fused) / nc * 100

        print(f"{nc:>6} | {t_old:>10.4f} | {t_fused:>10.4f} | {speedup:>7.2f}x | {scaling:>6.1f}%")

    # Restore threads
    set_num_threads(max_threads)

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    base_old = old_times[core_counts[0]]
    base_fused = fused_times[core_counts[0]]
    max_old = old_times[core_counts[-1]]
    max_fused = fused_times[core_counts[-1]]
    print(f"  Old path: {base_old:.4f}s ({core_counts[0]} cores) -> {max_old:.4f}s ({core_counts[-1]} cores) = {base_old/max_old:.1f}x scaling")
    print(f"  Fused:    {base_fused:.4f}s ({core_counts[0]} cores) -> {max_fused:.4f}s ({core_counts[-1]} cores) = {base_fused/max_fused:.1f}x scaling")
    print(f"  Fused speedup over old: {max_old / max_fused:.2f}x at {core_counts[-1]} cores")
    print("=" * 70)


if __name__ == "__main__":
    main()
