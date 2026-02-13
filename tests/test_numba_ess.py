"""
Verification test: compare new _calc_ess_numba against old calc_ESSs_vec + post-processing.

Run with:  python tests/test_numba_ess.py
"""
import sys
import os
# Ensure ESFS package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import scipy.sparse as spsparse
import time


def make_test_data(n_samples=500, n_fixed=50, n_features=200, density=0.3, seed=42):
    """Create random sparse test data mimicking ESFS inputs."""
    rng = np.random.default_rng(seed)

    # Create sparse matrices with values in [0, 1]
    ff_dense = rng.random((n_samples, n_fixed)) * (rng.random((n_samples, n_fixed)) < density)
    gs_dense = rng.random((n_samples, n_features)) * (rng.random((n_samples, n_features)) < density)

    fixed_features = spsparse.csc_matrix(ff_dense)
    fixed_features.sort_indices()
    global_scaled = spsparse.csc_matrix(gs_dense)
    global_scaled.sort_indices()

    sample_cardinality = n_samples

    # Feature sums and cardinalities
    ff_cardinality = np.asarray(fixed_features.sum(axis=0)).flatten()
    gs_sums = np.asarray(global_scaled.sum(axis=0)).flatten()

    # Minority states
    ff_minority = ff_cardinality.copy()
    ff_minority[ff_minority >= sample_cardinality / 2] = sample_cardinality - ff_minority[ff_minority >= sample_cardinality / 2]

    gs_minority = gs_sums.copy()
    gs_minority[gs_minority >= sample_cardinality / 2] = sample_cardinality - gs_minority[gs_minority >= sample_cardinality / 2]

    # FF_QF_vs_RF
    FF_QF_vs_RF = ff_minority[:, None] > gs_minority[None, :]

    # RFms, QFms, RFMs, QFMs
    RFms = np.where(~FF_QF_vs_RF, ff_minority[:, None], gs_minority[None, :])
    QFms = np.where(FF_QF_vs_RF, ff_minority[:, None], gs_minority[None, :])
    RFMs = sample_cardinality - RFms
    QFMs = sample_cardinality - QFms

    # max_ent_options
    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = np.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])

    return (fixed_features, global_scaled, sample_cardinality,
            RFms, QFms, RFMs, QFMs, max_ent_options,
            ff_cardinality, gs_sums, FF_QF_vs_RF)


def compute_overlaps_and_cases(fixed_features, global_scaled, sample_cardinality,
                                ff_cardinality, gs_sums, FF_QF_vs_RF):
    """Compute overlaps and case lookup tables."""
    from esfs.ESFS import overlaps_and_inverse_sparse

    ff_csc = fixed_features.tocsc()
    gs_csc = global_scaled.tocsc()

    target_dtype = np.float32
    ff_data = np.ascontiguousarray(ff_csc.data, dtype=target_dtype)
    ff_indices = np.ascontiguousarray(ff_csc.indices, dtype=np.int32)
    ff_indptr = np.ascontiguousarray(ff_csc.indptr, dtype=np.int32)
    gs_data = np.ascontiguousarray(gs_csc.data, dtype=target_dtype)
    gs_indices = np.ascontiguousarray(gs_csc.indices, dtype=np.int32)
    gs_indptr = np.ascontiguousarray(gs_csc.indptr, dtype=np.int32)

    overlaps, inverse_overlaps = overlaps_and_inverse_sparse(
        ff_data, ff_indices, ff_indptr,
        gs_data, gs_indices, gs_indptr,
        fixed_features.shape[0], fixed_features.shape[1], global_scaled.shape[1],
        use_float64=False
    )

    n_fixed = fixed_features.shape[1]
    n_features = global_scaled.shape[1]

    # Build case_idxs, case_patterns, overlap_lookup (matching get_overlap_info_vec)
    ff_is_min = (ff_cardinality < (sample_cardinality / 2))[:, None]
    sf_is_min = (gs_sums < (sample_cardinality / 2))[None, :]

    case_patterns = np.array([
        [0, -1, 0, 1],
        [0, 0, -1, 1],
        [-1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [1, 0, -1, 0],
        [1, -1, 0, 0],
    ], dtype=np.int8)

    case_idxs = (
        (ff_is_min.astype(int) << 2) + (sf_is_min.astype(int) << 1) + FF_QF_vs_RF.astype(int)
    ).astype(np.int8)

    row_map = np.stack(
        [np.argmax(case_patterns, axis=1), np.argmin(case_patterns, axis=1)],
        axis=1,
    ).astype(np.int8)

    overlap_lookup = np.full((8, 4), -1, dtype=np.int8)
    np.put_along_axis(overlap_lookup, row_map, [0, 1], axis=1)

    return overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup


def run_old_path(RFms, QFms, RFMs, QFMs, max_ent_options, sample_cardinality,
                 overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup):
    """Run the old vectorized path: calc_ESSs_vec + post-processing."""
    from esfs.ESFS import calc_ESSs_vec, nanmaximum

    ESSs, D_EPs, O_EPs, SWs, SGs = calc_ESSs_vec(
        RFms, QFms, RFMs, QFMs, max_ent_options,
        sample_cardinality, overlaps, inverse_overlaps,
        case_idxs, case_patterns, overlap_lookup,
        xp_mod=np,
    )
    # Post-processing
    iden_feats, iden_cols = np.nonzero(ESSs == 1)
    D_EPs[iden_feats, iden_cols] = 0
    O_EPs[iden_feats, iden_cols] = 0
    EPs = nanmaximum(D_EPs, O_EPs)
    return ESSs, EPs, SWs, SGs


def run_new_path(RFms, QFms, RFMs, QFMs, max_ent_options, sample_cardinality,
                 overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup):
    """Run the new Numba kernel path (Phase 1: separate overlaps + ESS)."""
    from esfs.ESFS import _calc_ess_numba

    ESSs, EPs, SWs, SGs = _calc_ess_numba(
        RFms, QFms, RFMs, QFMs, max_ent_options,
        sample_cardinality, overlaps, inverse_overlaps,
        case_idxs, case_patterns, overlap_lookup,
    )
    return ESSs, EPs, SWs, SGs


def run_fused_path(fixed_features, global_scaled, sample_cardinality,
                   RFms, QFms, RFMs, QFMs, max_ent_options,
                   ff_cardinality, gs_sums, FF_QF_vs_RF):
    """Run the fused overlap+ESS kernel (Phase 2)."""
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


def compare_results(old, new, names, rtol=1e-5, atol=1e-7):
    """Compare old vs new results, allowing for NaN matching."""
    all_pass = True
    for old_arr, new_arr, name in zip(old, new, names):
        # Both NaN in same positions
        old_nan = np.isnan(old_arr)
        new_nan = np.isnan(new_arr)
        nan_match = np.array_equal(old_nan, new_nan)

        if not nan_match:
            n_mismatch = np.sum(old_nan != new_nan)
            print(f"  {name}: NaN pattern mismatch in {n_mismatch} positions")
            # Show some examples
            diff_mask = old_nan != new_nan
            idxs = np.argwhere(diff_mask)[:5]
            for idx in idxs:
                i, j = idx
                print(f"    [{i},{j}] old={'NaN' if old_nan[i,j] else old_arr[i,j]:.8f}, "
                      f"new={'NaN' if new_nan[i,j] else new_arr[i,j]:.8f}")
            all_pass = False
            continue

        # Compare non-NaN values
        valid = ~old_nan
        if not valid.any():
            print(f"  {name}: all NaN (OK)")
            continue

        old_valid = old_arr[valid].astype(np.float64)
        new_valid = new_arr[valid].astype(np.float64)

        close = np.allclose(old_valid, new_valid, rtol=rtol, atol=atol)
        if close:
            max_diff = np.max(np.abs(old_valid - new_valid))
            print(f"  {name}: PASS (max diff = {max_diff:.2e})")
        else:
            diffs = np.abs(old_valid - new_valid)
            max_diff = np.max(diffs)
            mean_diff = np.mean(diffs)
            n_bad = np.sum(~np.isclose(old_valid, new_valid, rtol=rtol, atol=atol))
            print(f"  {name}: FAIL ({n_bad} mismatches, max diff = {max_diff:.2e}, mean diff = {mean_diff:.2e})")
            # Show worst mismatches
            worst = np.argsort(diffs)[-5:]
            all_valid_idxs = np.argwhere(valid)
            for w in worst:
                i, j = all_valid_idxs[w]
                print(f"    [{i},{j}] old={old_valid[w]:.10f}, new={new_valid[w]:.10f}, diff={diffs[w]:.2e}")
            all_pass = False

    return all_pass


def main():
    print("=" * 60)
    print("Verification: _calc_ess_numba vs calc_ESSs_vec")
    print("=" * 60)

    # Generate test data
    print("\nGenerating test data...")
    (fixed_features, global_scaled, sample_cardinality,
     RFms, QFms, RFMs, QFMs, max_ent_options,
     ff_cardinality, gs_sums, FF_QF_vs_RF) = make_test_data(
        n_samples=500, n_fixed=50, n_features=200, density=0.3, seed=42
    )

    print(f"  Fixed features: {fixed_features.shape}")
    print(f"  Global scaled: {global_scaled.shape}")
    print(f"  Sample cardinality: {sample_cardinality}")

    # Compute overlaps
    print("\nComputing overlaps...")
    overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup = compute_overlaps_and_cases(
        fixed_features, global_scaled, sample_cardinality,
        ff_cardinality, gs_sums, FF_QF_vs_RF
    )

    # Warmup JIT
    print("\nWarming up Numba JIT...")
    small_args = (
        RFms[:2, :2], QFms[:2, :2], RFMs[:2, :2], QFMs[:2, :2],
        max_ent_options[:, :2, :2], sample_cardinality,
        overlaps[:2, :2], inverse_overlaps[:2, :2],
        case_idxs[:2, :2], case_patterns, overlap_lookup
    )
    run_new_path(*small_args)
    print("  JIT warmup complete.")

    # Run old path
    print("\nRunning old vectorized path...")
    t0 = time.perf_counter()
    old_results = run_old_path(
        RFms, QFms, RFMs, QFMs, max_ent_options, sample_cardinality,
        overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup
    )
    t_old = time.perf_counter() - t0
    print(f"  Time: {t_old:.4f}s")

    # Run new Numba path
    print("\nRunning new Numba path...")
    t0 = time.perf_counter()
    new_results = run_new_path(
        RFms, QFms, RFMs, QFMs, max_ent_options, sample_cardinality,
        overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup
    )
    t_new = time.perf_counter() - t0
    print(f"  Time: {t_new:.4f}s")

    # Compare
    print("\nComparing results:")
    names = ["ESSs", "EPs", "SWs", "SGs"]
    all_pass = compare_results(old_results, new_results, names)

    # --- Fused kernel (Phase 2) ---
    print("\n" + "-" * 40)
    print("Fused overlap+ESS kernel (Phase 2)")
    print("-" * 40)

    # Warmup fused kernel
    print("\nWarming up fused kernel JIT...")
    small_ff = spsparse.csc_matrix(fixed_features[:, :2].toarray()[:10, :])
    small_gs = spsparse.csc_matrix(global_scaled[:, :2].toarray()[:10, :])
    small_ff.sort_indices()
    small_gs.sort_indices()
    run_fused_path(small_ff, small_gs, sample_cardinality,
                   RFms[:2, :2], QFms[:2, :2], RFMs[:2, :2], QFMs[:2, :2],
                   max_ent_options[:, :2, :2], ff_cardinality[:2], gs_sums[:2], FF_QF_vs_RF[:2, :2])
    print("  Fused JIT warmup complete.")

    print("\nRunning fused kernel...")
    t0 = time.perf_counter()
    fused_results = run_fused_path(
        fixed_features, global_scaled, sample_cardinality,
        RFms, QFms, RFMs, QFMs, max_ent_options,
        ff_cardinality, gs_sums, FF_QF_vs_RF
    )
    t_fused = time.perf_counter() - t0
    print(f"  Time: {t_fused:.4f}s")

    print("\nComparing fused vs old (relaxed tol for float32 overlap precision):")
    fused_pass = compare_results(old_results, fused_results, names, rtol=1e-4, atol=1e-6)
    all_pass = all_pass and fused_pass

    if all_pass:
        print("\nAll checks PASSED!")
    else:
        print("\nSome checks FAILED - investigate differences.")

    # Run with larger data for timing
    print("\n" + "=" * 60)
    print("Performance test (larger dataset)")
    print("=" * 60)

    (fixed_features2, global_scaled2, sc2,
     RFms2, QFms2, RFMs2, QFMs2, meo2,
     ffc2, gss2, ffqr2) = make_test_data(
        n_samples=2000, n_fixed=200, n_features=500, density=0.2, seed=123
    )

    overlaps2, inverse_overlaps2, ci2, cp2, ol2 = compute_overlaps_and_cases(
        fixed_features2, global_scaled2, sc2, ffc2, gss2, ffqr2
    )

    print(f"\nFixed features: {fixed_features2.shape}, Global scaled: {global_scaled2.shape}")

    # Time the old overlap computation separately
    t0 = time.perf_counter()
    compute_overlaps_and_cases(fixed_features2, global_scaled2, sc2, ffc2, gss2, ffqr2)
    t_overlaps = time.perf_counter() - t0

    # Old vectorized (ESS only, overlaps already computed)
    t0 = time.perf_counter()
    run_old_path(RFms2, QFms2, RFMs2, QFMs2, meo2, sc2,
                 overlaps2, inverse_overlaps2, ci2, cp2, ol2)
    t_old_ess = time.perf_counter() - t0

    t_old_total = t_overlaps + t_old_ess
    print(f"\n  Old overlap:       {t_overlaps:.4f}s")
    print(f"  Old ESS:           {t_old_ess:.4f}s")
    print(f"  Old total:         {t_old_total:.4f}s")

    # New Numba ESS (overlaps pre-computed)
    t0 = time.perf_counter()
    run_new_path(RFms2, QFms2, RFMs2, QFMs2, meo2, sc2,
                 overlaps2, inverse_overlaps2, ci2, cp2, ol2)
    t_new_ess = time.perf_counter() - t0
    t_new_total = t_overlaps + t_new_ess
    print(f"\n  Phase 1 (overlap + Numba ESS): {t_new_total:.4f}s  ({t_old_total / t_new_total:.1f}x)")

    # Fused overlap + ESS
    t0 = time.perf_counter()
    run_fused_path(fixed_features2, global_scaled2, sc2,
                   RFms2, QFms2, RFMs2, QFMs2, meo2,
                   ffc2, gss2, ffqr2)
    t_fused2 = time.perf_counter() - t0
    print(f"  Phase 2 (fused overlap+ESS):   {t_fused2:.4f}s  ({t_old_total / t_fused2:.1f}x)")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
