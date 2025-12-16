###### ESFS ######

### Dependencies ###
from functools import partial
import os
from types import ModuleType
from typing import Optional
import warnings

import anndata as ad
import multiprocess
from numba import njit, prange
import numpy as np
import pandas as pd
from pathos.pools import ProcessPool
from p_tqdm import p_map
import scipy.sparse as spsparse
from tqdm import tqdm

from .backend import backend

xp = backend.xp
xpsparse = backend.xpsparse
USING_GPU = backend.using_gpu

###### Entropy Sorting (ES) metric calculations ######

# Using the Entropy Sorting (ES) mathematical framework, we may caclulate the ESS, EP, SW and SG correlation metrics
# outlined in Radley et al. 2023 for any pair of features. The below code takes an anndata object as an input, calculates the
# requested ES metrics against each variable/feature in the adata object, and adds them as an attribute to the object for later use.


def create_scaled_matrix(adata, clip_percentile=97.5, log_scale=False):
    """
    Prior to calculates ES metrics, the data will be scaled to have values
    between 0 and 1.
    """
    # Filter genes with no expression
    # NOTE: Using numpy easier here, it isn't performance-critical code
    keep_genes = adata.var_names[np.nonzero(adata.X.getnnz(axis=0) > 50)[0]]
    if keep_genes.shape[0] < adata.shape[1]:
        print(
            str(adata.shape[1] - keep_genes.shape[0])
            + " genes show no expression. Removing them from adata object"
        )
        adata = adata[:, keep_genes]
    # Convert to CSC sparse matrix for processing using appropriate backend
    scaled_expressions = _convert_sparse_array(adata.X.copy())
    if log_scale:
        scaled_expressions.data = xp.log2(scaled_expressions.data + 1)
    # Iterate through each gene
    n_rows, n_cols = scaled_expressions.shape
    upper = xp.zeros(n_cols, dtype=xp.float64)
    for col_idx in range(n_cols):
        start_idx = scaled_expressions.indptr[col_idx]
        end_idx = scaled_expressions.indptr[col_idx + 1]
        col_data = scaled_expressions.data[start_idx:end_idx]
        # Ensure 0s included in percentile calculation
        n_nonzero = len(col_data)
        n_zeros = n_rows - n_nonzero
        if n_nonzero > 0:
            # Quick check if percentile would be zero
            target_rank = (clip_percentile / 100) * n_rows
            if target_rank <= n_zeros:
                # It would be 0, so we take the max and continue
                upper[col_idx] = xp.max(col_data)
                continue
            # Lazily pad with zeros to replicate full column
            # NOTE: If memory becomes an issue, can adjust percentile based on sparsity
            col_data = xp.concatenate((xp.zeros(n_zeros, dtype=col_data.dtype), col_data))
            upper[col_idx] = xp.percentile(col_data, clip_percentile)
            if upper[col_idx] == 0:
                upper[col_idx] = xp.max(col_data)
        # Fallback to avoid division by zero
        else:
            upper[col_idx] = 1.0
    # Build a mapping from data index to column index
    col_indices = xp.zeros(len(scaled_expressions.data), dtype=xp.int32)
    for col_idxs in range(n_cols):
        start_idx = scaled_expressions.indptr[col_idxs]
        end_idx = scaled_expressions.indptr[col_idxs + 1]
        col_indices[start_idx:end_idx] = col_idxs
    upper_broadcast = upper[col_indices]
    # Clip and scale
    scaled_expressions.data = xp.minimum(scaled_expressions.data, upper_broadcast)
    scaled_expressions.data = scaled_expressions.data / upper_broadcast
    # Downcast to float32 when storing
    adata.layers["Scaled_Counts"] = scaled_expressions.astype(xp.float32)
    return adata

def _convert_sparse_array(arr):
    # If scipy sparse (specifically)
    if spsparse.issparse(arr):
        # If cupy used, appropriately convert from scipy
        if USING_GPU:
            if spsparse.isspmatrix_csr(arr):
                arr = xpsparse.csr_matrix(arr)
            elif spsparse.isspmatrix_csc(arr):
                arr = xpsparse.csc_matrix(arr)
            elif spsparse.isspmatrix_coo(arr):
                arr = xpsparse.coo_matrix(arr)
            elif spsparse.isspmatrix_dia(arr):
                arr = xpsparse.dia_matrix(arr)
        # No matter what, convert to CSC for processing
        arr = arr.tocsc()
    # If not sparse, convert to sparse with whatever backend is being used
    else:
        arr = xpsparse.csc_matrix(arr)
    return arr

def parallel_calc_es_matrices(
    adata,
    secondary_features_label="Self",
    save_matrices: tuple = ("ESSs", "EPs"),
    use_cores=-1,
    chunksize: Optional[int] = None,
):
    """
    Using the Entropy Sorting (ES) mathematical framework, we may caclulate the ESS, EP, SW and SG correlation metrics
    outlined in Radley et al. 2023.

    This function assumes that the user aims has a set of input features (secondary_features_label) that will be used to pairwise calculate ES metrics
    against each variable (column) of the provided anndata object. When secondary_features_label is left blank, it defaults
    to "Self" meaning ES metrics will be calcualted pairwise between all of the variables in adata. If secondary_features_label
    is not "Self", the user must point the algorithm to an attribute of adata that contains an array with the same number of samples
    as adata and a set of secondary features (columns).

    `save_matrices` disctates which ES metrics will be written to the outputted adata object. The options are "ESSs", "EPs", "SGs", "SWs".

    `use_cores` defines how many CPU cores to use. Is use_cores = -1, the software will use N-1 the number of cores available on the machine.

    `chunksize` is either used to chunk ESS calculation when using GPU acceleration, or passed to pathos when using CPU parallelisation.
    """
    ## Establish which secondary_features will be compared against each of the features in adata
    global secondary_features
    if secondary_features_label == "Self":
        secondary_features = adata.layers["Scaled_Counts"]
    else:
        print(
            "You have provided a 'secondary_features_label', implying that in the anndata object there is a corresponding csc_sparse martix object with rows as samples and columns as features. Each feature will be used to calculate ES scores for each of the variables of the adata object"
        )
        secondary_features = adata.obsm[secondary_features_label]
        # Ensure sparse csc matrix with appropriate backend
        secondary_features = _convert_sparse_array(secondary_features)
    #
    ## Create the global global_scaled_matrix array for faster parallel computing calculations
    global global_scaled_matrix
    global_scaled_matrix = adata.layers["Scaled_Counts"]
    ## Extract sample and feature cardinality
    sample_cardinality = global_scaled_matrix.shape[0]
    ## Calculate feature sums and minority states for each adata feature
    # NOTE: .A is needed here for scipy, but incompatible with cupy.
    global feature_sums
    if not USING_GPU:
        feature_sums = global_scaled_matrix.sum(axis=0).A.flatten()
    else:
        feature_sums = global_scaled_matrix.sum(axis=0).flatten()
    global minority_states
    minority_states = feature_sums.copy()
    idxs = xp.where(minority_states >= (sample_cardinality / 2))[0]
    minority_states[idxs] = sample_cardinality - minority_states[idxs]
    ####
    ## Provide indicies for parallel computing.
    feature_inds = xp.arange(secondary_features.shape[1])
    # Get number of cores to use
    use_cores = get_num_cores(use_cores)
    ## Perform calculations
    print("Calculating ESS and EP matricies.")
    print(
        "If progress bar freezes consider increasing system memory or reducing number of cores used with the 'use_cores' parameter as you may have hit a memory ceiling for your machine."
    )
    ## Parallel compute
    with np.errstate(divide="ignore", invalid="ignore"):
        # Use our GPU-accelerated vectorised function if possible
        if USING_GPU:
            results = calc_es_metrics_vec(
                feature_inds,
                sample_cardinality=sample_cardinality,
                feature_sums=feature_sums,
                minority_states=minority_states,
                chunksize=chunksize,
            )
        elif use_cores == 1:
            # Single core: run sequentially for easier debugging (only)
            warnings.warn(
                "Running ESFS in single-core mode. This is not recommended for large datasets as it will be slow. Set use_cores to -1 or a positive integer to parallelize."
            )
            results = [
                calc_es_metrics(
                    ind,
                    sample_cardinality=sample_cardinality,
                )
                for ind in tqdm(feature_inds)
            ]
        else:
            # Multi-core: use parallel processing
            with ProcessPool(nodes=use_cores) as pool:
                results = []
                for res in tqdm(
                    pool.imap(
                        partial(calc_es_metrics, sample_cardinality=sample_cardinality),
                        feature_inds,
                        chunksize=chunksize if chunksize is not None else 1, # Default chunksize to 1 if not provided
                    ),
                    total=len(feature_inds),
                ):
                    results.append(res)
                pool.clear()
    ## Unpack results
    results = xp.asarray(results)
    # NOTE: GPU/vectorised code gives (4, samples, samples) shape, while original CPU gives (samples, 4, samples), so align
    if USING_GPU:
        results = xp.moveaxis(results, 0, 1)
    ## Save outputs requested by the save_matrices paramater
    if "ESSs" in save_matrices:
        ESSs = results[:, 0, :]
        if (
            secondary_features_label == "Self"
        ):  ## The vast majority of outputs are symmetric, but float errors appear to make some non-symmetric. If we can fix this that could be cool.
            ESSs = ensure_symmetric(ESSs)
        #
        # Label_ESSs = pd.DataFrame(ESSs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[secondary_features_label + "_ESSs"] = ESSs.T
        print(
            "ESSs for "
            + secondary_features_label
            + " label have been saved to "
            + "'adata.varm['"
            + secondary_features_label
            + "_ESSs']'"
        )
        #
    if "EPs" in save_matrices:
        EPs = results[:, 1, :]
        if secondary_features_label == "Self":
            EPs = ensure_symmetric(EPs)
        #
        # Label_EPs = pd.DataFrame(EPs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[secondary_features_label + "_EPs"] = EPs.T
        print(
            "EPs for "
            + secondary_features_label
            + " label have been saved to "
            + "'adata.varm['"
            + secondary_features_label
            + "_EPs']'"
        )
        #
    if "SWs" in save_matrices:
        SWs = results[:, 2, :]
        if secondary_features_label == "Self":
            SWs = ensure_symmetric(SWs)
        #
        # Label_SWs = pd.DataFrame(SWs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[secondary_features_label + "_SWs"] = SWs.T
        print(
            "SWs for "
            + secondary_features_label
            + " label have been saved to "
            + "'adata.varm['"
            + secondary_features_label
            + "_SWs']'"
        )
        #
    if "SGs" in save_matrices:
        SGs = results[:, 3, :]
        if secondary_features_label == "Self":
            SGs = ensure_symmetric(SGs)
        #
        # Label_SGs = pd.DataFrame(SGs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[secondary_features_label + "_SGs"] = SGs.T
        print(
            "SGs for "
            + secondary_features_label
            + " label have been saved to "
            + "'adata.varm['"
            + secondary_features_label
            + "_SGs']'"
        )
    return adata

def get_num_cores(use_cores: int):
    # Grab number of cores available
    cores_avail = multiprocess.cpu_count()
    # Special check if we're running in a SLURM environment, where CPU count does not match CPUs allocated
    if "SLURM_CPUS_ON_NODE" in os.environ:
        cores_avail = min(cores_avail, int(os.environ["SLURM_CPUS_ON_NODE"]))
    print("Cores Available: " + str(cores_avail))
    # If user sets -1, use all avail but one core (arbitrary buffer)
    if use_cores == -1:
        use_cores = max(cores_avail - 1, 1)
    print("Cores Used: " + str(use_cores))
    return use_cores

def nanmaximum(arr1, arr2):
    """
    Element-wise maximum of two arrays, ignoring NaN values and converting inf to -inf.

    Parameters:
        arr1 (ndarray): First input array.
        arr2 (ndarray): Second input array.

    Returns:
        ndarray: Element-wise maximum of arr1 and arr2, ignoring NaN values.
    """
    # Replace all inf and -inf values with -inf for both arrays
    arr1[xp.isinf(arr1)] = -xp.inf
    arr2[xp.isinf(arr2)] = -xp.inf
    # Replace NaN values with -infinity for comparison
    arr1_nan = xp.isnan(arr1)
    arr2_nan = xp.isnan(arr2)
    arr1[arr1_nan] = -xp.inf
    arr2[arr2_nan] = -xp.inf
    # Compute the element-wise maximum
    result = xp.maximum(arr1, arr2)
    # Where both values are NaN, the result should be NaN
    nan_mask = arr1_nan & arr2_nan
    result[nan_mask] = xp.nan
    return result

def ensure_symmetric(arr):
    if USING_GPU:
        return (arr + arr.T) / 2
    else:
        return nanmaximum(arr, arr.T)

def calc_es_metrics(feature_ind, sample_cardinality):
    """
    This function calcualtes the ES metrics for one of the features in the secondary_features object against
    every variable/feature in the adata object.

    `feature_ind` - Indicates the column of secondary_features being used.
    `sample_cardinality` - Inherited scalar of the number of samples in adata.
    `feature_sums` - Inherited vector of the columns sums of adata.
    `minority_states` - Inherited vector of the minority state sums of each column of adata.
    """
    ## Extract the Fixed Feature (FF)
    fixed_feature = secondary_features[:, feature_ind].A
    fixed_feature_cardinality = xp.sum(fixed_feature)
    fixed_feature_minority_state = fixed_feature_cardinality
    if fixed_feature_minority_state >= (sample_cardinality / 2):
        fixed_feature_minority_state = sample_cardinality - fixed_feature_minority_state
    ## Identify where FF is the QF or RF
    FF_QF_vs_RF = xp.zeros(feature_sums.shape[0])
    FF_QF_vs_RF[xp.nonzero(fixed_feature_minority_state > minority_states)[0]] = (
        1  # 1's mean FF is QF
    )
    ## Caclulate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = minority_states.copy()
    idxs = xp.where(FF_QF_vs_RF == 0)[0]
    RFms[idxs] = fixed_feature_minority_state
    RFMs = sample_cardinality - RFms
    QFms = minority_states.copy()
    idxs = xp.where(FF_QF_vs_RF == 1)[0]
    QFms[idxs] = fixed_feature_minority_state
    QFMs = sample_cardinality - QFms
    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = xp.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])
    ######
    ## Caclulate the overlap between the FF states and the secondary features, using the correct ESE (1-4)
    all_use_cases, all_overlaps_options, all_used_inds = get_overlap_info(
        fixed_feature,
        fixed_feature_cardinality,
        sample_cardinality,
        feature_sums,
        FF_QF_vs_RF,
    )
    ## Having extracted the overlaps and their respective ESEs (1-4), calcualte the ESS and EPs
    ESSs, D_EPs, O_EPs, SWs, SGs = calc_ESSs_old(
        RFms,
        QFms,
        RFMs,
        QFMs,
        max_ent_options,
        sample_cardinality,
        all_overlaps_options,
        all_use_cases,
        all_used_inds,
    )
    identical_features = xp.nonzero(ESSs == 1)[0]
    D_EPs[identical_features] = 0
    O_EPs[identical_features] = 0
    EPs = nanmaximum(D_EPs, O_EPs)
    return ESSs, EPs, SWs, SGs


def calc_es_metrics_vec(
    feature_inds, sample_cardinality, feature_sums, minority_states, num_cores=-1, chunksize: Optional[int] = None
):
    """
    This function calcualtes the ES metrics for one of the features in the secondary_features object against
    every variable/feature in the adata object.

    `feature_ind` - Indicates the column of secondary_features being used.
    `sample_cardinality` - Inherited scalar of the number of samples in adata.
    `feature_sums` - Inherited vector of the columns sums of adata.
    `minority_states` - Inherited vector of the minority state sums of each column of adata.
    """
    ## Extract the Fixed Feature (FF)
    fixed_features = secondary_features[:, feature_inds]
    if xpsparse.issparse(fixed_features):
        if USING_GPU:
            fixed_features_cardinality = fixed_features.sum(axis=0).flatten()
        else:
            fixed_features_cardinality = fixed_features.sum(axis=0).A.flatten()
    else:
        fixed_features_cardinality = fixed_features.toarray().sum(axis=0)
    fixed_feature_minority_states = fixed_features_cardinality.copy()
    idxs = fixed_feature_minority_states >= (sample_cardinality / 2)
    if idxs.any():
        fixed_feature_minority_states[idxs] = (
            sample_cardinality - fixed_feature_minority_states[idxs]
        )
    ## Identify where FF is the QF or RF
    FF_QF_vs_RF = fixed_feature_minority_states[:, None] > minority_states[None, :]
    ## Calculate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = xp.where(~FF_QF_vs_RF, fixed_feature_minority_states[:, None], minority_states[None, :])
    QFms = xp.where(FF_QF_vs_RF, fixed_feature_minority_states[:, None], minority_states[None, :])
    RFMs = sample_cardinality - RFms
    QFMs = sample_cardinality - QFms

    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = xp.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])
    ######
    ## Calculate the overlap between the FF states and the secondary features
    all_ESSs = []
    all_EPs = []
    all_SWs = []
    all_SGs = []
    overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup = get_overlap_info_vec(
        fixed_features,
        fixed_features_cardinality,
        sample_cardinality,
        feature_sums,
        FF_QF_vs_RF,
        njobs=None if num_cores == -1 else num_cores,
    )

    all_ESSs, all_D_EPs, all_O_EPs, all_SWs, all_SGs = calc_ESSs_chunked(
        RFms,
        QFms,
        RFMs,
        QFMs,
        max_ent_options,
        sample_cardinality,
        overlaps,
        inverse_overlaps,
        case_idxs,
        case_patterns,
        overlap_lookup,
        xp_mod=xp,
        chunksize=chunksize,
    )
    iden_feats, iden_cols = xp.nonzero(all_ESSs == 1)
    all_D_EPs[iden_feats, iden_cols] = 0
    all_O_EPs[iden_feats, iden_cols] = 0
    all_EPs = nanmaximum(all_D_EPs, all_O_EPs)
    return all_ESSs, all_EPs, all_SWs, all_SGs


def get_overlap_info(
    fixed_feature,
    fixed_feature_cardinality,
    sample_cardinality,
    feature_sums,
    FF_QF_vs_RF,
):
    """
    For any pair of features the ES mathematical framework has a set of logical rules regarding how the ES metrics
    should be calcualted. These logical rules dictate which of the two features is the reference feature (RF) or query feature
    (QF) and which of the 4 Entropy Sort Equations (ESE 1-4) should be used (Add reference to supplemental figure of
    new manuscript when ready).
    """
    ## Set up an array to track which of ESE equations 1-4 the recorded observed overlap relates to (row), and if it is
    # native correlation (1) or flipped anti-correlation (-1). Row 1 = mm, row 2 = Mm, row 3 = mM, row 4 = MM.
    all_use_cases = xp.zeros((4, feature_sums.shape[0]))
    ## Set up an array to track the observed overlaps between the FF and the secondary features.
    all_overlaps_options = xp.zeros((4, feature_sums.shape[0]))
    ## Set up a list to track the used inds/features for each ESE
    all_used_inds = [[] for _ in range(4)]
    ####
    ## Pairwise calculate total overlaps of FF values with the values every other a feature in adata
    nonzero_inds = xp.where(fixed_feature != 0)[0]
    sub_global_scaled_matrix = global_scaled_matrix[nonzero_inds, :]
    if (sub_global_scaled_matrix.indices.shape[0] > 0) and (nonzero_inds.shape[0] > 0):
        B = xpsparse.csc_matrix(
            (
                fixed_feature[nonzero_inds].T[0][sub_global_scaled_matrix.indices],
                sub_global_scaled_matrix.indices,
                sub_global_scaled_matrix.indptr,
            )
        )
        if USING_GPU:
            overlaps = sub_global_scaled_matrix.minimum(B.tocsr()).sum(axis=0)[0]
        else:
            overlaps = sub_global_scaled_matrix.minimum(B).sum(axis=0).A[0]
    else:
        overlaps = xp.zeros(global_scaled_matrix.shape[1])
    ## Pairwise calculate total overlaps of Inverse FF values with the values every other a feature in adata
    inverse_fixed_feature = 1 - fixed_feature  # xp.max(fixed_feature) - fixed_feature
    nonzero_inds = xp.where(inverse_fixed_feature != 0)[0]
    sub_global_scaled_matrix = global_scaled_matrix[nonzero_inds, :]
    if (sub_global_scaled_matrix.indices.shape[0] > 0) and (nonzero_inds.shape[0] > 0):
        B = xpsparse.csc_matrix(
            (
                inverse_fixed_feature[nonzero_inds].T[0][sub_global_scaled_matrix.indices],
                sub_global_scaled_matrix.indices,
                sub_global_scaled_matrix.indptr,
            )
        )
        if USING_GPU:
            inverse_overlaps = sub_global_scaled_matrix.minimum(B.tocsr()).sum(axis=0)[0]
        else:
            inverse_overlaps = sub_global_scaled_matrix.minimum(B).sum(axis=0).A[0]
    else:
        inverse_overlaps = xp.zeros(global_scaled_matrix.shape[1])
    ####
    ### Using the logical rules of ES to work out which ESE should be used for each pair of features beign compared.
    ## If FF is observed in it's minority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality < (sample_cardinality / 2):
        #######
        ## FF and other feature are minority states & FF is QF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as mm (row 1), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = xp.array([1, -1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        #######
        ## FF and other feature are minority states & FF is RF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as mm (row 1), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = xp.array([1, 0, -1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is QF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as mM (row 4), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = xp.array([0, 0, 1, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is RF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as Mm (row 2), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = xp.array([0, 1, 0, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        #
    ## If FF is observed in it's majority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality >= (sample_cardinality / 2):
        #######
        ## FF is majority, other feature is minority & FF is QF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as Mm (row 2), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = xp.array([-1, 1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is minority & FF is RF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as mM (row 3), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = xp.array([-1, 0, 1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is QF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as MM (row 4), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = xp.array([0, 0, -1, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is RF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as MM (row 4), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = xp.array([0, -1, 0, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        #
    return all_use_cases, all_overlaps_options, all_used_inds


def get_overlap_info_vec(
    fixed_features, fixed_features_cardinality, sample_cardinality, feature_sums, FF_QF_vs_RF, njobs
):
    """
    For any pair of features the ES mathematical framework has a set of logical rules regarding how the ES metrics
    should be calcualted. These logical rules dictate which of the two features is the reference feature (RF) or query feature
    (QF) and which of the 4 Entropy Sort Equations (ESE 1-4) should be used (Add reference to supplemental figure of
    new manuscript when ready).
    """
    ## Pairwise calculate total overlaps of FF values with the values every other a feature in adata
    if USING_GPU:
        # Convert to CSC format for column-wise access (if not already)
        fixed_features_csc = fixed_features.tocsc()
        global_scaled_matrix_csc = global_scaled_matrix.tocsc()
        overlaps = xp.zeros((fixed_features.shape[1], feature_sums.shape[0]), dtype=xp.float32)
        kernel = overlaps_sparse_cuda()
        block = (16, 16)
        grid = (
            (fixed_features.shape[1] + block[0] - 1) // block[0],
            (feature_sums.shape[0] + block[1] - 1) // block[1],
        )
        kernel(
            grid,
            block,
            (
                fixed_features_csc.data.astype(xp.float32),
                fixed_features_csc.indices,
                fixed_features_csc.indptr,
                global_scaled_matrix_csc.data.astype(xp.float32),
                global_scaled_matrix_csc.indices,
                global_scaled_matrix_csc.indptr,
                overlaps.ravel(),
                fixed_features.shape[1],
                feature_sums.shape[0],
            ),
        )

        # Now compute inverse overlaps using sparse-aware kernel
        inverse_overlaps = xp.zeros(
            (fixed_features.shape[1], feature_sums.shape[0]), dtype=xp.float32
        )
        kernel_inv = inverse_overlaps_sparse_cuda()
        kernel_inv(
            grid,
            block,
            (
                fixed_features_csc.data.astype(xp.float32),
                fixed_features_csc.indices,
                fixed_features_csc.indptr,
                global_scaled_matrix_csc.data.astype(xp.float32),
                global_scaled_matrix_csc.indices,
                global_scaled_matrix_csc.indptr,
                feature_sums.astype(xp.float32),
                inverse_overlaps.ravel(),
                fixed_features.shape[0],
                fixed_features.shape[1],
                feature_sums.shape[0],
            ),
        )
    else:
        overlaps = overlaps_cpu_parallel(
            fixed_features.toarray(),
            global_scaled_matrix.data,
            global_scaled_matrix.indices,
            global_scaled_matrix.indptr,
            fixed_features.shape[1],
            global_scaled_matrix.shape[1],
        )
        inverse_overlaps = overlaps_cpu_parallel(
            1 - fixed_features.toarray(),
            global_scaled_matrix.data,
            global_scaled_matrix.indices,
            global_scaled_matrix.indptr,
            fixed_features.shape[1],
            global_scaled_matrix.shape[1],
        )

    # Calculate our ineqs and reshape for broadcasting
    ff_is_min = (fixed_features_cardinality < (sample_cardinality / 2))[:, None]
    sf_is_min = (feature_sums < (sample_cardinality / 2))[None, :]
    # Map each case to its corresponding indices
    case_patterns = xp.array(
        [
            [0, -1, 0, 1],  # case_8: ff=0, sf=0, FF_QF_vs_RF=0
            [0, 0, -1, 1],  # case_7: ff=0, sf=0, FF_QF_vs_RF=1
            [-1, 0, 1, 0],  # case_6: ff=0, sf=1, FF_QF_vs_RF=0
            [-1, 1, 0, 0],  # case_5: ff=0, sf=1, FF_QF_vs_RF=1
            [0, 1, 0, -1],  # case_4: ff=1, sf=0, FF_QF_vs_RF=0
            [0, 0, 1, -1],  # case_3: ff=1, sf=0, FF_QF_vs_RF=1
            [1, 0, -1, 0],  # case_2: ff=1, sf=1, FF_QF_vs_RF=0
            [1, -1, 0, 0],  # case_1: ff=1, sf=1, FF_QF_vs_RF=1
        ],
        dtype=xp.int8,
    )
    # Shift to get 4/2/1 for each case, mapping to indices for the cases defined above
    # NOTE: These ints are the indices of the case_patterns array, NOT the case number, i.e. 7 = case_1
    case_idxs = (
        (ff_is_min.astype(int) << 2) + (sf_is_min.astype(int) << 1) + FF_QF_vs_RF.astype(int)
    ).astype(xp.int8)
    # Get the rows for each case where we insert the overlaps (+1) and inverse overlaps (-1)
    row_map = xp.stack(
        [xp.argmax(case_patterns, axis=1), xp.argmin(case_patterns, axis=1)],
        axis=1,
        dtype=xp.int8,
    )
    # Now define a lookup table for which overlap to use
    # 0 = overlaps, 1 = inverse_overlaps, -1 = not relevant
    overlap_lookup = xp.full((8, 4), -1, dtype=xp.int8)
    # Insert the overlaps and inverse overlaps for each case for each col (mm, Mm, mM, MM)
    # NOTE: put_along_axis is availabile in cupy v14, which is prerelease at the time of writing
    # So we just drop down to numpy and convert back to cupy until it's in stable release to avoid issues
    if USING_GPU:
        # Convert to numpy
        overlap_lookup = xp.asnumpy(overlap_lookup).astype(xp.int8)
        row_map = xp.asnumpy(row_map).astype(xp.int8)
        # Run using numpy explicitly
        np.put_along_axis(overlap_lookup, row_map, [0, 1], axis=1)
        # Convert back to cupy
        overlap_lookup = xp.asarray(overlap_lookup, dtype=xp.int8)
        row_map = xp.asarray(row_map, dtype=xp.int8)
    else:
        xp.put_along_axis(overlap_lookup, row_map, [0, 1], axis=1)
    # Now return the overlaps, and what we need to extract everything needed for ESS calcs later
    return overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup


def overlaps_sparse_cuda():
    """
    CUDA kernel for computing overlaps using sparse CSC format for fixed_features.
    This avoids the memory explosion from densifying sparse matrices.
    """
    kernel_code = r"""
    extern "C" __global__
    void compute_overlaps_sparse(const float* ff_data,
                                const int* ff_indices,
                                const int* ff_indptr,
                                const float* gs_data,
                                const int* gs_indices,
                                const int* gs_indptr,
                                float* overlaps,
                                int n_fixed_features,
                                int n_features) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;  // fixed_feature index
        int j = blockIdx.y * blockDim.y + threadIdx.y;  // feature index

        if (i >= n_fixed_features || j >= n_features) return;

        float sum = 0.0f;

        // Get sparse column ranges for both matrices
        int ff_start = ff_indptr[i];
        int ff_end = ff_indptr[i + 1];
        int gs_start = gs_indptr[j];
        int gs_end = gs_indptr[j + 1];

        // Two-pointer merge algorithm to find overlapping rows
        int ff_ptr = ff_start;
        int gs_ptr = gs_start;

        while (ff_ptr < ff_end && gs_ptr < gs_end) {
            int ff_row = ff_indices[ff_ptr];
            int gs_row = gs_indices[gs_ptr];

            if (ff_row == gs_row) {
                // Both matrices have values at this row
                float ff_val = ff_data[ff_ptr];
                float gs_val = gs_data[gs_ptr];
                sum += fminf(ff_val, gs_val);
                ff_ptr++;
                gs_ptr++;
            } else if (ff_row < gs_row) {
                ff_ptr++;
            } else {
                gs_ptr++;
            }
        }

        overlaps[i * n_features + j] = sum;
    }
    """

    module = xp.RawModule(code=kernel_code)
    return module.get_function("compute_overlaps_sparse")


def inverse_overlaps_sparse_cuda():
    """
    CUDA kernel for computing inverse overlaps: min(1-ff, gs).
    Uses sparse representation to avoid densification.
    Key insight: min(1-ff, gs) = gs - min(ff, gs) when ff > 0, or gs when ff = 0
    So we compute: sum_all_gs_nonzeros(gs) - sum_overlaps(ff, gs) + corrections
    """
    kernel_code = r"""
    extern "C" __global__
    void compute_inverse_overlaps_sparse(const float* ff_data,
                                        const int* ff_indices,
                                        const int* ff_indptr,
                                        const float* gs_data,
                                        const int* gs_indices,
                                        const int* gs_indptr,
                                        const float* gs_sums,
                                        float* inverse_overlaps,
                                        int n_samples,
                                        int n_fixed_features,
                                        int n_features) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;  // fixed_feature index
        int j = blockIdx.y * blockDim.y + threadIdx.y;  // feature index

        if (i >= n_fixed_features || j >= n_features) return;

        // Strategy: min(1-ff, gs) at each position
        // For positions where gs=0: contributes 0
        // For positions where gs>0, ff=0: contributes gs  (1-0=1, min(1,gs)=gs)
        // For positions where gs>0, ff>0: contributes min(1-ff, gs)

        float sum = 0.0f;

        int ff_start = ff_indptr[i];
        int ff_end = ff_indptr[i + 1];
        int gs_start = gs_indptr[j];
        int gs_end = gs_indptr[j + 1];

        int ff_ptr = ff_start;
        int gs_ptr = gs_start;

        // Iterate through gs nonzeros
        while (gs_ptr < gs_end) {
            int gs_row = gs_indices[gs_ptr];
            float gs_val = gs_data[gs_ptr];

            // Find if ff has a value at this row
            float ff_val = 0.0f;
            while (ff_ptr < ff_end && ff_indices[ff_ptr] < gs_row) {
                ff_ptr++;
            }

            if (ff_ptr < ff_end && ff_indices[ff_ptr] == gs_row) {
                ff_val = ff_data[ff_ptr];
            }

            // Compute min(1 - ff_val, gs_val)
            sum += fminf(1.0f - ff_val, gs_val);

            gs_ptr++;
        }

        inverse_overlaps[i * n_features + j] = sum;
    }
    """

    module = xp.RawModule(code=kernel_code)
    return module.get_function("compute_inverse_overlaps_sparse")

@njit(parallel=True)
def overlaps_cpu_parallel(fixed_features, data, indices, indptr, n_fixed_features, n_features):
    """
    Compute overlaps between fixed features and sparse matrix data.
    """
    # NOTE: To exactly recreate original code, this needs to be calculated into 32-bits
    # In the original code, this was then upcast to 64 later, but still resulted in 0s where expected
    # Calculating this in 64 bits directly will not lead to the same 0s in very very few edge cases
    overlaps = np.zeros((n_fixed_features, n_features), dtype=np.float32)
    # Parallelize over the outer loop (fixed features)
    for i in prange(n_fixed_features):
        for j in range(n_features):
            sum_val = 0.0
            start = indptr[j]
            end = indptr[j + 1]
            for k in range(start, end):
                row = indices[k]
                ff_val = fixed_features[row, i]
                if ff_val != 0.0:
                    gs_val = data[k]
                    sum_val += min(ff_val, gs_val)
            overlaps[i, j] = sum_val
    return overlaps

def calc_ESSs_chunked(
    RFms,
    QFms,
    RFMs,
    QFMs,
    max_ent_options,
    sample_cardinality,
    overlaps,
    inverse_overlaps,
    case_idxs,
    case_patterns,
    overlap_lookup,
    xp_mod: Optional[ModuleType] = None,
    chunksize: Optional[int] = None,
):
    if xp_mod is None:
        xp_mod = xp
    # If no chunksize is provided, process all at once
    if chunksize is None:
        all_ESSs, all_D_EPs, all_O_EPs, all_SWs, all_SGs = calc_ESSs_vec(
            RFms,
            QFms,
            RFMs,
            QFMs,
            max_ent_options,
            sample_cardinality,
            overlaps,
            inverse_overlaps,
            case_idxs,
            case_patterns,
            overlap_lookup,
            xp_mod=xp,
        )
        return all_ESSs, all_D_EPs, all_O_EPs, all_SWs, all_SGs
    # Otherwise, process in chunks
    n_feats = RFms.shape[0]
    n_comps = RFms.shape[1]
    final_ESSs = xp_mod.zeros((n_feats, n_comps), dtype=backend.dtype)
    final_D_EPs = xp_mod.zeros((n_feats, n_comps), dtype=backend.dtype)
    final_O_EPs = xp_mod.zeros((n_feats, n_comps), dtype=backend.dtype)
    final_SWs = xp_mod.zeros((n_feats, n_comps), dtype=backend.dtype)
    final_SGs = xp_mod.zeros((n_feats, n_comps), dtype=backend.dtype)

    for start_idx in tqdm(range(0, n_feats, chunksize), desc="ESS Chunks"):
        end_idx = min(start_idx + chunksize, n_feats)
        (
            chunk_ESSs,
            chunk_D_EPs,
            chunk_O_EPs,
            chunk_SWs,
            chunk_SGs,
        ) = calc_ESSs_vec(
            RFms[start_idx:end_idx],
            QFms[start_idx:end_idx],
            RFMs[start_idx:end_idx],
            QFMs[start_idx:end_idx],
            max_ent_options[:, start_idx:end_idx],
            sample_cardinality,
            overlaps[start_idx:end_idx],
            inverse_overlaps[start_idx:end_idx],
            case_idxs[start_idx:end_idx],
            case_patterns,
            overlap_lookup,
            xp_mod=xp_mod,
        )
        final_ESSs[start_idx:end_idx] = chunk_ESSs
        final_D_EPs[start_idx:end_idx] = chunk_D_EPs
        final_O_EPs[start_idx:end_idx] = chunk_O_EPs
        final_SWs[start_idx:end_idx] = chunk_SWs
        final_SGs[start_idx:end_idx] = chunk_SGs
    return final_ESSs, final_D_EPs, final_O_EPs, final_SWs, final_SGs

def calc_ESSs_vec(
    RFms,
    QFms,
    RFMs,
    QFMs,
    max_ent_options,
    sample_cardinality,
    overlaps,
    inverse_overlaps,
    case_idxs,
    case_patterns,
    overlap_lookup,
    xp_mod
):
    n_feats = RFms.shape[0]
    n_comps = RFms.shape[1]
    # Create output arrays, with NaNs for easier later masking/maximum selection
    # NOTE: Using float32 to save memory, should be sufficient precision
    # NOTE: If needed, we can cut the first dim and iteratively store the max
    all_ESSs = xp_mod.full((4, n_feats, n_comps), xp_mod.nan, dtype=backend.dtype)
    all_D_EPs = xp_mod.full((4, n_feats, n_comps), xp_mod.nan, dtype=backend.dtype)
    all_O_EPs = xp_mod.full((4, n_feats, n_comps), xp_mod.nan, dtype=backend.dtype)
    all_SGs = xp_mod.full((4, n_feats, n_comps), xp_mod.nan, dtype=backend.dtype)
    all_SWs = xp_mod.full((4, n_feats, n_comps), xp_mod.nan, dtype=backend.dtype)

    for use_curve in range(4):
        curve_mask = case_patterns[case_idxs, use_curve] != 0
        # Skip this curve if nothing to calculate
        if not xp_mod.any(curve_mask):
            continue
        overlap_source = overlap_lookup[case_idxs, use_curve]
        # NOTE: Not sure if we should be upcasting here
        curve_overlaps = xp_mod.where(overlap_source == 0, overlaps, inverse_overlaps).astype(
            xp_mod.float64
        )
        max_ent_x = max_ent_options[use_curve]
        SD_1_mask = (curve_overlaps < max_ent_x) & curve_mask
        SD1_mask = ~SD_1_mask & curve_mask
        # NOTE: Returns float64 as default, so ensure consistent
        SDs = xp_mod.where(SD1_mask, 1.0, -1.0).astype(backend.dtype)
        min_overlap, max_overlap = get_overlap_bounds_vec(
            use_curve, RFms=RFms, RFMs=RFMs, QFms=QFms, QFMs=QFMs, xp_mod=xp_mod
        )
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        D = xp_mod.zeros((n_feats, n_comps), dtype=backend.dtype)
        O = xp_mod.zeros((n_feats, n_comps), dtype=backend.dtype)  # noqa: E741
        if use_curve == 0:
            D = xp_mod.where(SD_1_mask, curve_overlaps, RFms - curve_overlaps)
            O = xp_mod.where(  # noqa: E741
                SD_1_mask,
                sample_cardinality - (RFms + QFms) + curve_overlaps,
                QFms - curve_overlaps,
            )
            CE, ind_E, min_E = ESE1_batched(
                curve_overlaps,
                SDs,
                RFms,
                RFMs,
                QFms,
                QFMs,
                sample_cardinality,
                max_overlap,
                curve_mask,
                xp_mod,
            )
        elif use_curve == 1:
            D = xp_mod.where(SD_1_mask, curve_overlaps, RFms - curve_overlaps)
            O = xp_mod.where(SD_1_mask, QFms - RFms + curve_overlaps, QFMs - RFms + D)  # noqa: E741
            CE, ind_E, min_E = ESE2_batched(
                curve_overlaps, SDs, RFms, RFMs, QFms, QFMs, sample_cardinality, curve_mask, xp_mod
            )
        elif use_curve == 2:
            D = xp_mod.where(SD_1_mask, QFMs - RFMs + curve_overlaps, QFms - curve_overlaps)
            O = xp_mod.where(SD_1_mask, curve_overlaps, RFMs - curve_overlaps)  # noqa: E741
            CE, ind_E, min_E = ESE3_batched(
                curve_overlaps,
                SDs,
                RFms,
                RFMs,
                QFms,
                QFMs,
                sample_cardinality,
                min_overlap,
                curve_mask,
                xp_mod,
            )
        elif use_curve == 3:
            D = xp_mod.where(
                SD_1_mask,
                curve_overlaps - (sample_cardinality - (QFms + RFms)),
                QFMs - curve_overlaps,
            )
            O = xp_mod.where(SD_1_mask, curve_overlaps, RFMs - QFMs + D)  # noqa: E741
            CE, ind_E, min_E = ESE4_batched(
                curve_overlaps,
                SDs,
                RFms,
                RFMs,
                QFms,
                QFMs,
                sample_cardinality,
                min_overlap,
                max_overlap,
                curve_mask,
                xp_mod,
            )
        all_ESSs, all_SWs, all_SGs, all_D_EPs, all_O_EPs = common_ES_metrics_batched(
            all_ESSs,
            all_SWs,
            all_SGs,
            all_D_EPs,
            all_O_EPs,
            ind_E,
            min_E,
            CE,
            SDs,
            D,
            O,
            ind_X_1,
            ind_X1,
            case_patterns,
            case_idxs,
            use_curve,
            SD_1_mask,
            SD1_mask,
            curve_mask,
            xp_mod,
        )
    # For each feature pair, accept the orientation with the maximum ESS as it is the least likely to have occurred by chance.
    max_ESS_idxs = xp_mod.nanargmax(xp_mod.absolute(all_ESSs), axis=0)
    # Gather results using advanced indexing
    feature_idx = xp_mod.arange(n_feats)[:, None]
    comparison_idx = xp_mod.arange(n_comps)[None, :]
    return (
        all_ESSs[max_ESS_idxs, feature_idx, comparison_idx],
        all_D_EPs[max_ESS_idxs, feature_idx, comparison_idx],
        all_O_EPs[max_ESS_idxs, feature_idx, comparison_idx],
        all_SWs[max_ESS_idxs, feature_idx, comparison_idx],
        all_SGs[max_ESS_idxs, feature_idx, comparison_idx],
    )

def get_overlap_bounds_vec(use_curve, RFms, RFMs, QFms, QFMs, xp_mod):
    # Return the min and max overlap values for the given curve
    if use_curve == 0:
        return xp_mod.zeros_like(RFms), RFms
    elif use_curve == 1:
        return xp_mod.zeros_like(RFms), xp_mod.minimum(RFms, QFMs)
    elif use_curve == 2:
        return RFMs - QFMs, xp_mod.minimum(QFms, RFMs)
    elif use_curve == 3:
        return QFMs - RFms, xp_mod.minimum(QFMs, RFMs)
    else:
        raise ValueError(f"Invalid use_curve value: {use_curve}")


def common_ES_metrics_batched(
    all_ESSs,
    all_SWs,
    all_SGs,
    all_D_EPs,
    all_O_EPs,
    ind_E,
    min_E,
    CE,
    SDs,
    D,
    O,  # noqa: E741
    ind_X_1,
    ind_X1,
    case_patterns,
    curve_case_idxs,
    use_curve,
    SD_1_mask,
    SD1_mask,
    curve_mask,
    xp_mod,
):
    """Vectorized version that handles all features at once with masking."""
    # Avoid division by zero by using where
    all_SWs[use_curve] = xp_mod.where(curve_mask & (ind_E != 0), (ind_E - min_E) / ind_E, all_SWs[use_curve])
    all_SGs[use_curve] = xp_mod.where(curve_mask & ((ind_E - min_E) != 0), (ind_E - CE) / (ind_E - min_E), all_SGs[use_curve])
    # Correct boundary float errors where not NaN
    all_SGs[use_curve] = xp_mod.where(xp_mod.isnan(all_SGs[use_curve]), xp_mod.nan, xp_mod.clip(all_SGs[use_curve], 0, 1))

    all_ESSs[use_curve] = xp_mod.where(curve_mask, all_SWs[use_curve] * all_SGs[use_curve] * SDs * case_patterns[curve_case_idxs, use_curve], all_ESSs[use_curve])
    # Calculate entropy per unit, avoiding division by zero
    SD_1_IndEnt = xp_mod.where(SD_1_mask & (ind_X_1 != 0), ind_E / ind_X_1, 0)
    SD1_IndEnt = xp_mod.where(SD1_mask & (ind_X1 != 0), ind_E / ind_X1, 0)

    # Calculate D_EPs
    all_D_EPs[use_curve] = xp_mod.where(SD_1_mask & (D != 0), ((CE - min_E) / D) - SD_1_IndEnt, all_D_EPs[use_curve])
    all_D_EPs[use_curve] = xp_mod.where(SD1_mask & (D != 0), ((CE - min_E) / D) - SD1_IndEnt, all_D_EPs[use_curve])
    # Calculate O_EPs
    all_O_EPs[use_curve] = xp_mod.where(SD_1_mask & (O != 0), (CE / O) - SD_1_IndEnt, all_O_EPs[use_curve])
    all_O_EPs[use_curve] = xp_mod.where(SD1_mask & (O != 0), (CE / O) - SD1_IndEnt, all_O_EPs[use_curve])
    return all_ESSs, all_SWs, all_SGs, all_D_EPs, all_O_EPs

def calc_ESSs_old(
    RFms,
    QFms,
    RFMs,
    QFMs,
    max_ent_options,
    sample_cardinality,
    all_overlaps_options,
    all_use_cases,
    all_used_inds,
):
    """
    Now that we have all of the values required for ES caclulations (RFms, QFms, RFMs, QFMs, max_ent_options) and
    have determined which ESE should be used for each pair of features (all_overlaps_options, all_use_cases, all_used_inds),
    we may calculated the ES metrics for the FF against every other feature in adata.
    """
    ## Create variables to track caclulation outputs
    all_ESSs = xp.zeros((4, RFms.shape[0]))
    all_D_EPs = xp.zeros((4, RFms.shape[0]))
    all_O_EPs = xp.zeros((4, RFms.shape[0]))
    all_SGs = xp.zeros((4, RFms.shape[0]))
    all_SWs = xp.zeros((4, RFms.shape[0]))
    ###################
    ##### (1)  mm #####
    use_curve = 0
    ## Find the FF/SF pairs where we should use ESE (1) to calculate entropies
    calc_idxs = all_used_inds[use_curve].astype("i")
    if calc_idxs.shape[0] > 0:
        # Retrieve the max_ent, Min_x, Max_X and observed overlap values
        min_overlap = xp.zeros_like(calc_idxs)
        max_overlap = RFms[calc_idxs]
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = xp.where(overlaps < max_ent_x)[0]
        SD1_idxs = xp.where(overlaps >= max_ent_x)[0]
        SDs = xp.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = xp.zeros(calc_idxs.shape[0])
        O = xp.zeros(calc_idxs.shape[0])  # noqa: E741
        D[SD_1_idxs] = overlaps[SD_1_idxs]
        O[SD_1_idxs] = (
            sample_cardinality
            - (RFms[calc_idxs][SD_1_idxs] + QFms[calc_idxs][SD_1_idxs])
            + overlaps[SD_1_idxs]
        )
        D[SD1_idxs] = RFms[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        O[SD1_idxs] = QFms[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        # Perform caclulations with ESE (1)
        CE, ind_E, min_E = ESE1(
            overlaps,
            SDs,
            RFms[calc_idxs],
            RFMs[calc_idxs],
            QFms[calc_idxs],
            QFMs[calc_idxs],
            sample_cardinality,
            max_overlap,
            xp_mod=xp,
        )
        #
        SWs = (ind_E - min_E) / ind_E
        SGs = (ind_E - CE) / (ind_E - min_E)
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * all_use_cases[use_curve, calc_idxs]
        all_ESSs[use_curve, calc_idxs] = ESS
        all_SWs[use_curve, calc_idxs] = SWs
        all_SGs[use_curve, calc_idxs] = SGs
        #
        SD_1_IndEnt = ind_E[SD_1_idxs] / ind_X_1[SD_1_idxs]
        SD1_IndEnt = ind_E[SD1_idxs] / ind_X1[SD1_idxs]
        #
        D_EPs = xp.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = ((CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = xp.zeros(ind_E.shape[0])
        O_EPs[SD_1_idxs] = ((CE[SD_1_idxs]) / O[SD_1_idxs]) - SD_1_IndEnt
        O_EPs[SD1_idxs] = ((CE[SD1_idxs]) / O[SD1_idxs]) - SD1_IndEnt
        #
        all_D_EPs[use_curve, calc_idxs] = D_EPs
        all_O_EPs[use_curve, calc_idxs] = O_EPs
        #
    ###################
    ##### (2)  Mm #####
    use_curve = 1
    ## Find the FF/SF pairs where we should use ESE (2) to calculate entropies
    calc_idxs = all_used_inds[use_curve].astype("i")
    if calc_idxs.shape[0] > 0:
        # Retrieve the max_ent, Min_x, Max_X and observed overlap values
        min_overlap = xp.zeros_like(calc_idxs)
        max_overlap = xp.minimum(RFms[calc_idxs], QFMs[calc_idxs])
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = xp.where(overlaps < max_ent_x)[0]
        SD1_idxs = xp.where(overlaps >= max_ent_x)[0]
        SDs = xp.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = xp.zeros(calc_idxs.shape[0])
        O = xp.zeros(calc_idxs.shape[0])  # noqa: E741
        D[SD_1_idxs] = overlaps[SD_1_idxs]
        O[SD_1_idxs] = QFms[calc_idxs][SD_1_idxs] - RFms[calc_idxs][SD_1_idxs] + overlaps[SD_1_idxs]
        D[SD1_idxs] = RFms[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        O[SD1_idxs] = QFMs[calc_idxs][SD1_idxs] - RFms[calc_idxs][SD1_idxs] + D[SD1_idxs]
        # Perform caclulations with ESE (2)
        CE, ind_E, min_E = ESE2(
            overlaps,
            SDs,
            RFms[calc_idxs],
            RFMs[calc_idxs],
            QFms[calc_idxs],
            QFMs[calc_idxs],
            sample_cardinality,
            xp_mod=xp,
        )
        #
        SWs = (ind_E - min_E) / ind_E
        SGs = (ind_E - CE) / (ind_E - min_E)
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * all_use_cases[use_curve, calc_idxs]
        all_ESSs[use_curve, calc_idxs] = ESS
        all_SWs[use_curve, calc_idxs] = SWs
        all_SGs[use_curve, calc_idxs] = SGs
        #
        SD_1_IndEnt = ind_E[SD_1_idxs] / ind_X_1[SD_1_idxs]
        SD1_IndEnt = ind_E[SD1_idxs] / ind_X1[SD1_idxs]
        #
        D_EPs = xp.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = ((CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = xp.zeros(ind_E.shape[0])
        O_EPs[SD_1_idxs] = ((CE[SD_1_idxs]) / O[SD_1_idxs]) - SD_1_IndEnt
        O_EPs[SD1_idxs] = ((CE[SD1_idxs]) / O[SD1_idxs]) - SD1_IndEnt
        #
        all_D_EPs[use_curve, calc_idxs] = D_EPs
        all_O_EPs[use_curve, calc_idxs] = O_EPs
        #
    ###################
    ##### (3)  mM #####
    use_curve = 2
    ## Find the FF/SF pairs where we should use ESE (3) to calculate entropies
    calc_idxs = all_used_inds[use_curve].astype("i")
    if calc_idxs.shape[0] > 0:
        # Retrieve the max_ent, Min_x, Max_X and observed overlap values
        min_overlap = RFMs[calc_idxs] - QFMs[calc_idxs]
        max_overlap = xp.minimum(QFms[calc_idxs], RFMs[calc_idxs])
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = xp.where(overlaps < max_ent_x)[0]
        SD1_idxs = xp.where(overlaps >= max_ent_x)[0]
        SDs = xp.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = xp.zeros(calc_idxs.shape[0])
        O = xp.zeros(calc_idxs.shape[0])  # noqa: E741
        D[SD_1_idxs] = QFMs[calc_idxs][SD_1_idxs] - RFMs[calc_idxs][SD_1_idxs] + overlaps[SD_1_idxs]
        O[SD_1_idxs] = overlaps[SD_1_idxs]
        D[SD1_idxs] = QFms[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        O[SD1_idxs] = RFMs[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        # Perform caclulations with ESE (3)
        CE, ind_E, min_E = ESE3(
            overlaps,
            SDs,
            RFms[calc_idxs],
            RFMs[calc_idxs],
            QFms[calc_idxs],
            QFMs[calc_idxs],
            sample_cardinality,
            min_overlap,
            xp_mod=xp,
        )
        #
        SWs = (ind_E - min_E) / ind_E
        SGs = (ind_E - CE) / (ind_E - min_E)
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * all_use_cases[use_curve, calc_idxs]
        all_ESSs[use_curve, calc_idxs] = ESS
        all_SWs[use_curve, calc_idxs] = SWs
        all_SGs[use_curve, calc_idxs] = SGs
        #
        SD_1_IndEnt = ind_E[SD_1_idxs] / ind_X_1[SD_1_idxs]
        SD1_IndEnt = ind_E[SD1_idxs] / ind_X1[SD1_idxs]
        #
        D_EPs = xp.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = ((CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = xp.zeros(ind_E.shape[0])
        O_EPs[SD_1_idxs] = ((CE[SD_1_idxs]) / O[SD_1_idxs]) - SD_1_IndEnt
        O_EPs[SD1_idxs] = ((CE[SD1_idxs]) / O[SD1_idxs]) - SD1_IndEnt
        #
        all_D_EPs[use_curve, calc_idxs] = D_EPs
        all_O_EPs[use_curve, calc_idxs] = O_EPs
        #
    ###################
    ##### (4)  MM #####
    use_curve = 3
    ## Find the FF/SF pairs where we should use ESE (4) to calculate entropies
    calc_idxs = all_used_inds[use_curve].astype("i")
    if calc_idxs.shape[0] > 0:
        # Retrieve the max_ent, Min_x, Max_X and observed overlap values
        min_overlap = QFMs[calc_idxs] - RFms[calc_idxs]
        max_overlap = xp.minimum(QFMs[calc_idxs], RFMs[calc_idxs])
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = xp.where(overlaps < max_ent_x)[0]
        SD1_idxs = xp.where(overlaps >= max_ent_x)[0]
        SDs = xp.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = xp.zeros(calc_idxs.shape[0])
        O = xp.zeros(calc_idxs.shape[0])  # noqa: E741
        D[SD_1_idxs] = overlaps[SD_1_idxs] - (
            sample_cardinality - (QFms[calc_idxs][SD_1_idxs] + RFms[calc_idxs][SD_1_idxs])
        )
        O[SD_1_idxs] = overlaps[SD_1_idxs]
        D[SD1_idxs] = QFMs[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        O[SD1_idxs] = RFMs[calc_idxs][SD1_idxs] - QFMs[calc_idxs][SD1_idxs] + D[SD1_idxs]
        # Perform caclulations with ESE (4)
        CE, ind_E, min_E = ESE4(
            overlaps,
            SDs,
            RFms[calc_idxs],
            RFMs[calc_idxs],
            QFms[calc_idxs],
            QFMs[calc_idxs],
            sample_cardinality,
            min_overlap,
            max_overlap,
            xp_mod=xp,
        )
        #
        SWs = (ind_E - min_E) / ind_E
        SGs = (ind_E - CE) / (ind_E - min_E)
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * all_use_cases[use_curve, calc_idxs]
        all_ESSs[use_curve, calc_idxs] = ESS
        all_SWs[use_curve, calc_idxs] = SWs
        all_SGs[use_curve, calc_idxs] = SGs
        #
        SD_1_IndEnt = ind_E[SD_1_idxs] / ind_X_1[SD_1_idxs]
        SD1_IndEnt = ind_E[SD1_idxs] / ind_X1[SD1_idxs]
        #
        D_EPs = xp.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = ((CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = xp.zeros(ind_E.shape[0])
        O_EPs[SD_1_idxs] = ((CE[SD_1_idxs]) / O[SD_1_idxs]) - SD_1_IndEnt
        O_EPs[SD1_idxs] = ((CE[SD1_idxs]) / O[SD1_idxs]) - SD1_IndEnt
        #
        all_D_EPs[use_curve, calc_idxs] = D_EPs
        all_O_EPs[use_curve, calc_idxs] = O_EPs
        #
    ########
    ## For each feature pair, accept the orientation with the maximum ESS as it is the least likely to have occoured by chance.
    max_ESS_idxs = xp.nanargmax(xp.absolute(all_ESSs), axis=0)
    ## Return results
    return (
        all_ESSs[max_ESS_idxs, xp.arange(RFms.shape[0])],
        all_D_EPs[max_ESS_idxs, xp.arange(RFms.shape[0])],
        all_O_EPs[max_ESS_idxs, xp.arange(RFms.shape[0])],
        all_SWs[max_ESS_idxs, xp.arange(RFms.shape[0])],
        all_SGs[max_ESS_idxs, xp.arange(RFms.shape[0])],
    )


def ESE1(x, SD, RFm, RFM, QFm, QFM, Ts, max_overlap, xp_mod):
    """
    This function takes the observed inputs and uses the ESE1 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (((x) / RFm) * (-xp_mod.log((x) / RFm)))
        + (((RFm - x) / RFm) * (-xp_mod.log((RFm - x) / RFm)))
    )
    G2_E = (RFM / Ts) * (
        (((QFm - x) / RFM) * (-xp_mod.log((QFm - x) / RFM)))
        + (((RFM - QFm + x) / RFM) * (-xp_mod.log((RFM - QFm + x) / RFM)))
    )
    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))
    #
    min_E = xp_mod.zeros(SD.shape[0])
    SD_1_idxs = xp_mod.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (((QFm[SD_1_idxs]) / RFM[SD_1_idxs]) * (-xp_mod.log((QFm[SD_1_idxs]) / RFM[SD_1_idxs])))
        + (
            ((RFM[SD_1_idxs] - QFm[SD_1_idxs]) / RFM[SD_1_idxs])
            * (-xp_mod.log((RFM[SD_1_idxs] - QFm[SD_1_idxs]) / RFM[SD_1_idxs]))
        )
    )
    SD1_idxs = xp_mod.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            ((QFm[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            * (-xp_mod.log((QFm[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs]))
        )
        + (
            ((RFM[SD1_idxs] - QFm[SD1_idxs] + max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            * (-xp_mod.log((RFM[SD1_idxs] - QFm[SD1_idxs] + max_overlap[SD1_idxs]) / RFM[SD1_idxs]))
        )
    )
    min_E[xp_mod.isnan(min_E)] = 0
    #
    CE[xp_mod.isnan(CE)] = min_E[xp_mod.isnan(CE)]
    return CE, ind_E, min_E


def ESE2(x, SD, RFm, RFM, QFm, QFM, Ts, xp_mod):
    """
    This function takes the observed inputs and uses the ESE2 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (-(((RFm - x) / RFm) * xp_mod.log((RFm - x) / RFm)) - (((x) / RFm) * xp_mod.log((x) / RFm)))
    )
    G2_E = (RFM / Ts) * (
        (
            -(((RFM - QFM + x) / RFM) * xp_mod.log((RFM - QFM + x) / RFM))
            - (((QFM - x) / RFM) * xp_mod.log((QFM - x) / RFM))
        )
    )
    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))
    #
    min_E = xp_mod.zeros(SD.shape[0])
    SD_1_idxs = xp_mod.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD_1_idxs] - QFM[SD_1_idxs]) / RFM[SD_1_idxs])
                * xp_mod.log((RFM[SD_1_idxs] - QFM[SD_1_idxs]) / RFM[SD_1_idxs])
            )
            - (((QFM[SD_1_idxs]) / RFM[SD_1_idxs]) * xp_mod.log((QFM[SD_1_idxs]) / RFM[SD_1_idxs]))
        )
    )
    SD1_idxs = xp_mod.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs])
                * xp_mod.log((RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs])
            )
            - (
                ((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
                * xp_mod.log((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
            )
        )
    )
    min_E[xp_mod.isnan(min_E)] = 0
    #
    CE[xp_mod.isnan(CE)] = min_E[xp_mod.isnan(CE)]
    return CE, ind_E, min_E


def ESE3(x, SD, RFm, RFM, QFm, QFM, Ts, min_overlap, xp_mod):
    """
    This function takes the observed inputs and uses the ESE3 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (
            -(((QFm - x) / RFm) * xp_mod.log((QFm - x) / RFm))
            - (((RFm - QFm + x) / RFm) * xp_mod.log((RFm - QFm + x) / RFm))
        )
    )
    G2_E = (RFM / Ts) * (
        (-(((x) / RFM) * xp_mod.log((x) / RFM)) - (((RFM - x) / RFM) * xp_mod.log((RFM - x) / RFM)))
    )
    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))
    #
    min_E = xp_mod.zeros(SD.shape[0])
    SD_1_idxs = xp_mod.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (
            -(
                ((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * xp_mod.log((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
            - (
                ((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * xp_mod.log((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
        )
    )
    SD1_idxs = xp_mod.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs])
                * xp_mod.log((RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs])
            )
            - (
                ((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
                * xp_mod.log((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
            )
        )
    )
    min_E[xp_mod.isnan(min_E)] = 0
    #
    CE[xp_mod.isnan(CE)] = min_E[xp_mod.isnan(CE)]
    return CE, ind_E, min_E


def ESE4(x, SD, RFm, RFM, QFm, QFM, Ts, min_overlap, max_overlap, xp_mod):
    """
    This function takes the observed inputs and uses the ESE4 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (
            -(((RFm - QFM + x) / RFm) * xp_mod.log((RFm - QFM + x) / RFm))
            - (((QFM - x) / RFm) * xp_mod.log((QFM - x) / RFm))
        )
    )
    G2_E = (RFM / Ts) * (
        (-(((RFM - x) / RFM) * xp_mod.log((RFM - x) / RFM)) - (((x) / RFM) * xp_mod.log((x) / RFM)))
    )
    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))
    #
    min_E = xp_mod.zeros(SD.shape[0])
    SD_1_idxs = xp_mod.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * xp_mod.log((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
            - (
                ((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * xp_mod.log((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
        )
    )
    SD1_idxs = xp_mod.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs])
                * xp_mod.log((RFM[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            )
            - (
                ((max_overlap[SD1_idxs]) / RFM[SD1_idxs])
                * xp_mod.log((max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            )
        )
    )
    min_E[xp_mod.isnan(min_E)] = 0
    #
    CE[xp_mod.isnan(CE)] = min_E[xp_mod.isnan(CE)]
    return CE, ind_E, min_E


def ESE1_batched(x, SD, RFm, RFM, QFm, QFM, Ts, max_overlap, mask, xp_mod):
    """
    Batched version of ESE1 that processes all features simultaneously.
    All inputs are 2D arrays of shape (n_features, n_comparisons).
    """
    # Calculate G1_E and G2_E (fully vectorized)
    G1_E = (RFm / Ts) * (
        (((x) / RFm) * (-xp_mod.log((x) / RFm)))
        + (((RFm - x) / RFm) * (-xp_mod.log((RFm - x) / RFm)))
    )
    G2_E = (RFM / Ts) * (
        (((QFm - x) / RFM) * (-xp_mod.log((QFm - x) / RFM)))
        + (((RFM - QFm + x) / RFM) * (-xp_mod.log((RFM - QFm + x) / RFM)))
    )

    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))

    # Calculate min_E using masks instead of index arrays
    min_E = xp_mod.zeros_like(SD)

    # For SD == -1
    SD_1_mask = SD == -1
    min_E_SD_1 = (RFM / Ts) * (
        (((QFm) / RFM) * (-xp_mod.log((QFm) / RFM)))
        + (((RFM - QFm) / RFM) * (-xp_mod.log((RFM - QFm) / RFM)))
    )
    min_E = xp_mod.where(SD_1_mask, min_E_SD_1, min_E)

    # For SD == 1
    SD1_mask = SD == 1
    min_E_SD1 = (RFM / Ts) * (
        (((QFm - max_overlap) / RFM) * (-xp_mod.log((QFm - max_overlap) / RFM)))
        + (((RFM - QFm + max_overlap) / RFM) * (-xp_mod.log((RFM - QFm + max_overlap) / RFM)))
    )
    min_E = xp_mod.where(SD1_mask, min_E_SD1, min_E)

    # Handle NaN values
    min_E = xp_mod.where(xp_mod.isnan(min_E), 0, min_E)
    CE = xp_mod.where(xp_mod.isnan(CE), min_E, CE)

    # Apply mask and NaN out invalid calcs
    CE = xp_mod.where(mask, CE, xp_mod.nan)
    ind_E = xp_mod.where(mask, ind_E, xp_mod.nan)
    min_E = xp_mod.where(mask, min_E, xp_mod.nan)

    return CE, ind_E, min_E


def ESE2_batched(x, SD, RFm, RFM, QFm, QFM, Ts, mask, xp_mod):
    """
    Batched version of ESE2 that processes all features simultaneously.
    All inputs are 2D arrays of shape (n_features, n_comparisons).
    """
    # Calculate G1_E and G2_E (fully vectorized)
    G1_E = (RFm / Ts) * (
        (-(((RFm - x) / RFm) * xp_mod.log((RFm - x) / RFm)) - (((x) / RFm) * xp_mod.log((x) / RFm)))
    )
    G2_E = (RFM / Ts) * (
        (
            -(((RFM - QFM + x) / RFM) * xp_mod.log((RFM - QFM + x) / RFM))
            - (((QFM - x) / RFM) * xp_mod.log((QFM - x) / RFM))
        )
    )

    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))

    # Calculate min_E using masks
    min_E = xp_mod.zeros_like(SD)

    # For SD == -1
    SD_1_mask = SD == -1
    min_E_SD_1 = (RFM / Ts) * (
        (
            -(((RFM - QFM) / RFM) * xp_mod.log((RFM - QFM) / RFM))
            - (((QFM) / RFM) * xp_mod.log((QFM) / RFM))
        )
    )
    min_E = xp_mod.where(SD_1_mask, min_E_SD_1, min_E)

    # For SD == 1
    SD1_mask = SD == 1
    min_E_SD1 = (RFM / Ts) * (
        (
            -(((RFM - QFM + RFm) / RFM) * xp_mod.log((RFM - QFM + RFm) / RFM))
            - (((QFM - RFm) / RFM) * xp_mod.log((QFM - RFm) / RFM))
        )
    )
    min_E = xp_mod.where(SD1_mask, min_E_SD1, min_E)

    # Handle NaN values
    min_E = xp_mod.where(xp_mod.isnan(min_E), 0, min_E)
    CE = xp_mod.where(xp_mod.isnan(CE), min_E, CE)

    # Apply mask and NaN out invalid calcs
    CE = xp_mod.where(mask, CE, xp_mod.nan)
    ind_E = xp_mod.where(mask, ind_E, xp_mod.nan)
    min_E = xp_mod.where(mask, min_E, xp_mod.nan)

    return CE, ind_E, min_E


def ESE3_batched(x, SD, RFm, RFM, QFm, QFM, Ts, min_overlap, mask, xp_mod):
    """
    Batched version of ESE3 that processes all features simultaneously.
    All inputs are 2D arrays of shape (n_features, n_comparisons).
    """
    # Calculate G1_E and G2_E (fully vectorized)
    G1_E = (RFm / Ts) * (
        (
            -(((QFm - x) / RFm) * xp_mod.log((QFm - x) / RFm))
            - (((RFm - QFm + x) / RFm) * xp_mod.log((RFm - QFm + x) / RFm))
        )
    )
    G2_E = (RFM / Ts) * (
        (-(((x) / RFM) * xp_mod.log((x) / RFM)) - (((RFM - x) / RFM) * xp_mod.log((RFM - x) / RFM)))
    )

    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))

    # Calculate min_E using masks
    min_E = xp_mod.zeros_like(SD)

    # For SD == -1
    SD_1_mask = SD == -1
    min_E_SD_1 = (RFM / Ts) * (
        (
            -(((min_overlap) / RFM) * xp_mod.log((min_overlap) / RFM))
            - (((RFM - min_overlap) / RFM) * xp_mod.log((RFM - min_overlap) / RFM))
        )
    )
    min_E = xp_mod.where(SD_1_mask, min_E_SD_1, min_E)

    # For SD == 1
    SD1_mask = SD == 1
    min_E_SD1 = (RFM / Ts) * (
        (
            -(((RFM - QFM + RFm) / RFM) * xp_mod.log((RFM - QFM + RFm) / RFM))
            - (((QFM - RFm) / RFM) * xp_mod.log((QFM - RFm) / RFM))
        )
    )
    min_E = xp_mod.where(SD1_mask, min_E_SD1, min_E)

    # Handle NaN values
    min_E = xp_mod.where(xp_mod.isnan(min_E), 0, min_E)
    CE = xp_mod.where(xp_mod.isnan(CE), min_E, CE)

    # Apply mask and NaN out invalid calcs
    CE = xp_mod.where(mask, CE, xp_mod.nan)
    ind_E = xp_mod.where(mask, ind_E, xp_mod.nan)
    min_E = xp_mod.where(mask, min_E, xp_mod.nan)

    return CE, ind_E, min_E


def ESE4_batched(x, SD, RFm, RFM, QFm, QFM, Ts, min_overlap, max_overlap, mask, xp_mod):
    """
    Batched version of ESE4 that processes all features simultaneously.
    All inputs are 2D arrays of shape (n_features, n_comparisons).
    """
    # Calculate G1_E and G2_E (fully vectorized)
    G1_E = (RFm / Ts) * (
        (
            -(((RFm - QFM + x) / RFm) * xp_mod.log((RFm - QFM + x) / RFm))
            - (((QFM - x) / RFm) * xp_mod.log((QFM - x) / RFm))
        )
    )
    G2_E = (RFM / Ts) * (
        (-(((RFM - x) / RFM) * xp_mod.log((RFM - x) / RFM)) - (((x) / RFM) * xp_mod.log((x) / RFM)))
    )

    CE = xp_mod.where(xp_mod.isnan(G1_E), 0, G1_E) + xp_mod.where(xp_mod.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-xp_mod.log((QFm / Ts))) + (QFM / Ts) * (-xp_mod.log((QFM / Ts)))

    # Calculate min_E using masks
    min_E = xp_mod.zeros_like(SD)

    # For SD == -1
    SD_1_mask = SD == -1
    min_E_SD_1 = (RFM / Ts) * (
        (
            -(((RFM - min_overlap) / RFM) * xp_mod.log((RFM - min_overlap) / RFM))
            - (((min_overlap) / RFM) * xp_mod.log((min_overlap) / RFM))
        )
    )
    min_E = xp_mod.where(SD_1_mask, min_E_SD_1, min_E)

    # For SD == 1
    SD1_mask = SD == 1
    min_E_SD1 = (RFM / Ts) * (
        (
            -(((RFM - max_overlap) / RFM) * xp_mod.log((RFM - max_overlap) / RFM))
            - (((max_overlap) / RFM) * xp_mod.log((max_overlap) / RFM))
        )
    )
    min_E = xp_mod.where(SD1_mask, min_E_SD1, min_E)

    # Handle NaN values
    min_E = xp_mod.where(xp_mod.isnan(min_E), 0, min_E)
    CE = xp_mod.where(xp_mod.isnan(CE), min_E, CE)

    # Apply mask and NaN out invalid calcs
    CE = xp_mod.where(mask, CE, xp_mod.nan)
    ind_E = xp_mod.where(mask, ind_E, xp_mod.nan)
    min_E = xp_mod.where(mask, min_E, xp_mod.nan)

    return CE, ind_E, min_E


##### Combinatorial cluster marker gene identification functions #####

## For any dataset, we may cluster the samples into groups. We may then be interested in which features best describe different groupings
# of the data. However, we may not know what resolution of the data best describes the patterns we are interested in. This ambuguity may
# be remedied by looking at every combination of clusters for an intentionally overclustered dataset. However, searching every combination
# of clusters quickly becomes computationally intractible.  To overcome this challange, we introduce a combinatorial clustering algorithm
# that turns the the combinatorial problem into a linear one, which can be tractably solved in mamy practical scenarios.


def ES_CCF(adata, secondary_features_label, use_cores: int = -1, chunksize: Optional[int] = None):
    """
    This function takes an anndata object containing an attribute relating to a set of secondary_features and a attributes containing
    the ESS and SG Entropy Sorting metrics calculated pairwise for each feature secondary_features against each feature of the
    variables contained in the anndata object using the parallel_calc_es_matrices function. For combinatorial cluster marker
    gene identification, secondary_features is created using the intentionally over-clustered samples labels of the samples in
    adata and converting them into a 2D-array through one hot encoding. This function will then identify which combination of
    one-hotted clusters maximises the correlation with the expression profile of each gene and hence identifies where in the data
    a gene may be considered a marker gene without providing any prior knowledge. The fucntion then attaches the optimum cluster
    combination and it's correspoinding ESS for each feature to the anndata object, using secondary_features_label as an identifier.
    """
    ###
    ess_label = f"{secondary_features_label}_ESSs"
    if ess_label not in adata.varm.keys():
        raise ValueError(
            f"ESSs for {secondary_features_label} not found in adata.varm. Please run parallel_calc_es_matrices first."
        )
    sg_label = f"{secondary_features_label}_SGs"
    if sg_label not in adata.varm.keys():
        raise ValueError(
            f"SGs for {secondary_features_label} not found in adata.varm. Please run parallel_calc_es_matrices first."
        )
    ## Create the global global_scaled_matrix array for faster parallel computing calculations
    global global_scaled_matrix
    global_scaled_matrix = adata.layers["Scaled_Counts"]
    ### Extract the secondary_features object from adata
    secondary_features = adata.obsm[secondary_features_label]
    # Ensure sparse csc matrix with appropriate backend
    secondary_features = _convert_sparse_array(secondary_features)
    ### Extract the secondary_features ESSs from adata
    all_ESSs = adata.varm[secondary_features_label + "_ESSs"]
    initial_max_ESSs = xp.asarray(xp.max(all_ESSs, axis=1))
    max_ESSs = initial_max_ESSs.copy()
    ### Extract the secondary_features SGs from adata
    all_SGs = xp.asarray(adata.varm[secondary_features_label + "_SGs"]).copy()
    all_SGs[all_ESSs < 0] = (
        all_SGs[all_ESSs < 0] * -1
    )  # For ordering we need to include the SG sort directions which we can extract from the ESSs
    ### For each feature in adata, sort the SGs of the secondary_features from highest to lowest. This is the main step that allows
    # use to turn the intractible combinatorial problem into a tractable linear problem.
    sorted_SGs_idxs = xp.argsort(all_SGs)[:, ::-1]
    ### Parallel warpper function
    max_ESSs, max_EPs, top_score_columns_combinations, top_score_secondary_features = (
        parallel_identify_max_ESSs(secondary_features, sorted_SGs_idxs, use_cores=use_cores, chunksize=chunksize)
    )
    # Compile results
    combinatorial_label_info = xp.column_stack(
        [xp.array(max_ESSs), xp.array(max_EPs)]
    )
    if USING_GPU:
        combinatorial_label_info = combinatorial_label_info.get()
    combinatorial_label_info = pd.DataFrame(
        combinatorial_label_info, index=adata.var_names, columns=["Max_ESSs", "EPs"]
    )  # ,"top_score_columns_combinations"])
    ### Save Max_Combinatorial cluster information and scores
    adata.varm[secondary_features_label + "_Max_Combinatorial_ESSs"] = combinatorial_label_info
    print(
        "Max combinatorial ESSs for given sample labels has been saved as 'adata.varm['"
        + secondary_features_label
        + "_Max_Combinatorial_ESSs"
        + "']'"
    )
    ### Save the new features/clusters that maximise the ESS scores of each feature in adata
    adata.obsm[secondary_features_label + "_Max_ESS_Features"] = xpsparse.csc_matrix(
        top_score_secondary_features.astype("f")
    )
    print(
        "The features/clusters relating to each max_ESS have been saved in 'adata.obsm['"
        + secondary_features_label
        + "_Max_ESS_Features"
        + "']'"
    )
    return adata


def parallel_identify_max_ESSs(secondary_features, sorted_SGs_idxs, use_cores=-1, chunksize: Optional[int] = None,):
    """
    Parallelised version of identify_max_ESSs function
    """
    #
    feature_inds = xp.arange(sorted_SGs_idxs.shape[0])
    # Get number of cores to use
    use_cores = get_num_cores(use_cores)
    ## Perform calculations
    print("Calculating ESS and EP matricies.")
    print(
        "If progress bar freezes consider increasing system memory or reducing number of cores used with the 'use_cores' parameter as you may have hit a memory ceiling for your machine."
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        if USING_GPU:
            # Use vectorized version for GPU
            all_ESSs, all_EPs = identify_max_ESSs_vec(
                feature_inds,
                secondary_features,
                sorted_SGs_idxs,
                chunksize=chunksize
            )
            # Convert to list of results for compatibility with downstream code
            results = [(all_ESSs[i], all_EPs[i]) for i in range(len(feature_inds))]
        elif use_cores == 1:
            results = [
                identify_max_ESSs(FF_ind, secondary_features, sorted_SGs_idxs)
                for FF_ind in tqdm(feature_inds)
            ]
        else:
            with ProcessPool(nodes=use_cores) as pool:
                results = []
                for res in tqdm(
                    pool.imap(
                        partial(identify_max_ESSs, secondary_features=secondary_features, sorted_SGs_idxs=sorted_SGs_idxs),
                        feature_inds,
                        chunksize=chunksize if chunksize is not None else 1, # Default chunksize to 1 if not provided
                    ),
                    total=len(feature_inds),
                ):
                    results.append(res)
                pool.clear()
    ## Extract results for each feature in adata
    max_ESSs = xp.zeros(len(results)).astype("f")
    max_EPs = xp.zeros(len(results)).astype("f")
    top_score_columns_combinations = [[]] * len(results)
    top_score_secondary_features = xp.zeros((secondary_features.shape[0], feature_inds.shape[0]))
    for i in range(len(results)):
        ESSs = results[i][0]
        EPs = results[i][1]
        max_ESS_idx = xp.argmax(ESSs)
        max_ESSs[i] = ESSs[max_ESS_idx]
        max_EPs[i] = EPs[max_ESS_idx]
        top_score_columns_combinations[i] = sorted_SGs_idxs[i, :][
            xp.arange(0, max_ESS_idx + 1)
        ].tolist()
        if USING_GPU:
            top_score_secondary_features[:, i] = (
                secondary_features[:, top_score_columns_combinations[i]].sum(axis=1).reshape(-1)
            )
        else:
            top_score_secondary_features[:, i] = (
                secondary_features[:, top_score_columns_combinations[i]].sum(axis=1).A.reshape(-1)
            )
    ## Return results
    return (
        max_ESSs,
        max_EPs,
        top_score_columns_combinations,
        top_score_secondary_features,
    )


def identify_max_ESSs(FF_ind, secondary_features, sorted_SGs_idxs):
    """
    For each fixed feature (FF_ind) in adata, this function identifies which combination of secondary_features (the one hot clustering
    of the intentionally overclustered samples in adata) maximises the ESS of the the fixed feature, thereby giving us a coarse grain
    approximation of how to cluster the data without having to decide how many clusters we expect there to be in the data.
    """
    ## Extract the fixed feature from adata
    fixed_feature = global_scaled_matrix[:, FF_ind].A
    sample_cardinality = fixed_feature.shape[0]
    ## Remove the lowest rank cluster to avoid a potential cluster size being equal to the number of samples in the data.
    sort_order = xp.delete(sorted_SGs_idxs[FF_ind, :], -1)
    ## From the ordered one-hut clusters, take the cumulative row sums, thereby creating the set of linearly combined one-hot
    # clusters for which we will calculate the ESSs of the fixed feature against.
    secondary_features = xp.cumsum(secondary_features.A[:, sort_order], axis=1)
    secondary_features = xpsparse.csc_matrix(secondary_features.astype("f"))
    #### Calculate constants required for ES calculations
    SF_sums = secondary_features.A.sum(axis=0)
    SF_minority_states = SF_sums.copy()
    SF_minority_states[SF_minority_states >= (sample_cardinality / 2)] = (
        sample_cardinality - SF_minority_states[SF_minority_states >= (sample_cardinality / 2)]
    )
    ##
    fixed_feature = fixed_feature.reshape(sample_cardinality, 1)  # Might be superfluous
    ## Calculate feature sums
    fixed_feature_cardinality = xp.sum(fixed_feature)
    fixed_feature_minority_state = fixed_feature_cardinality.copy()
    if fixed_feature_minority_state >= (sample_cardinality / 2):
        fixed_feature_minority_state = sample_cardinality - fixed_feature_minority_state
    #
    ## Identify where FF is the QF or RF
    FF_QF_vs_RF = xp.zeros(SF_minority_states.shape[0])
    FF_QF_vs_RF[xp.where(fixed_feature_minority_state > SF_minority_states)[0]] = (
        1  # 1's mean FF is QF
    )
    ## Caclulate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = SF_minority_states.copy()
    idxs = xp.where(FF_QF_vs_RF == 0)[0]
    RFms[idxs] = fixed_feature_minority_state
    RFMs = sample_cardinality - RFms
    QFms = SF_minority_states.copy()
    idxs = xp.where(FF_QF_vs_RF == 1)[0]
    QFms[idxs] = fixed_feature_minority_state
    QFMs = sample_cardinality - QFms
    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = xp.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])
    ####
    # Caclulate the overlap between the FF states and the secondary features, using the correct ESE (1-4)
    all_use_cases, all_overlaps_options, all_used_inds = identify_max_ESSs_get_overlap_info(
        fixed_feature,
        fixed_feature_cardinality,
        sample_cardinality,
        SF_sums,
        FF_QF_vs_RF,
        secondary_features,
    )
    ## Having extracted the overlaps and their respective ESEs (1-4), calcualte the ESS and EPs
    ESSs, D_EPs, O_EPs, SWs, SGs = calc_ESSs_old(
        RFms,
        QFms,
        RFMs,
        QFMs,
        max_ent_options,
        sample_cardinality,
        all_overlaps_options,
        all_use_cases,
        all_used_inds,
    )
    # EPs = xp.maximum(D_EPs,O_EPs)
    identical_features = xp.where(ESSs == 1)[0]
    D_EPs[identical_features] = 0
    O_EPs[identical_features] = 0
    EPs = nanmaximum(D_EPs, O_EPs)
    return ESSs, EPs


def identify_max_ESSs_get_overlap_info(
    fixed_feature,
    fixed_feature_cardinality,
    sample_cardinality,
    feature_sums,
    FF_QF_vs_RF,
    secondary_features,
):
    """
    This function is an adapted version of get_overlap_info where the 1D fixed feature is a variable/column from adata which is being
    compared against a set of secondary features (instead of the reverse scenario in get_overlap_info). This was done because secondary_features
    has to be newly generated in the identify_max_ESSs function for each variable/column in adata.

    For any pair of features the ES mathematical framework has a set of logical rules regarding how the ES metrics
    should be calcualted. These logical rules dictate which of the two features is the reference feature (RF) or query feature
    (QF) and which of the 4 Entropy Sort Equations (ESE 1-4) should be used (Add reference to supplemental figure of
    new manuscript when ready).
    """
    ## Set up an array to track which of ESE equations 1-4 the recorded observed overlap relates to (row), and if it is
    # native correlation (1) or flipped anti-correlation (-1). Row 1 = mm, row 2 = Mm, row 3 = mM, row 4 = MM.
    all_use_cases = xp.zeros((4, feature_sums.shape[0]))
    ## Set up an array to track the observed overlaps between the FF and the secondary features.
    all_overlaps_options = xp.zeros((4, feature_sums.shape[0]))
    ## Set up a list to track the used inds/features for each ESE
    all_used_inds = [[]] * 4
    #
    nonzero_inds = xp.where(fixed_feature != 0)[0]
    sub_secondary_features = secondary_features[nonzero_inds, :]
    B = xpsparse.csc_matrix(
        (
            fixed_feature[nonzero_inds].T[0][sub_secondary_features.indices],
            sub_secondary_features.indices,
            sub_secondary_features.indptr,
        )
    )
    if USING_GPU:
        overlaps = sub_secondary_features.minimum(B.tocsr()).sum(axis=0)[0]
    else:
        overlaps = sub_secondary_features.minimum(B).sum(axis=0).A[0]
    #
    inverse_fixed_feature = xp.max(fixed_feature) - fixed_feature
    nonzero_inds = xp.where(inverse_fixed_feature != 0)[0]
    sub_secondary_features = secondary_features[nonzero_inds, :]
    B = xpsparse.csc_matrix(
        (
            inverse_fixed_feature[nonzero_inds].T[0][sub_secondary_features.indices],
            sub_secondary_features.indices,
            sub_secondary_features.indptr,
        )
    )
    if USING_GPU:
        inverse_overlaps = sub_secondary_features.minimum(B.tocsr()).sum(axis=0)[0]
    else:
        inverse_overlaps = sub_secondary_features.minimum(B).sum(axis=0).A[0]
    ## If FF is observed in it's minority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality < (sample_cardinality / 2):
        #######
        ## FF and other feature are minority states & FF is QF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as mm (row 1), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = xp.array([1, -1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        #######
        ## FF and other feature are minority states & FF is RF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as mm (row 1), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = xp.array([1, 0, -1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is QF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as mM (row 4), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = xp.array([0, 0, 1, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is RF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as Mm (row 2), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = xp.array([0, 1, 0, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        #
    ## If FF is observed in it's majority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality >= (sample_cardinality / 2):
        #######
        ## FF is majority, other feature is minority & FF is QF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as Mm (row 2), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = xp.array([-1, 1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is minority & FF is RF
        calc_idxs = xp.where((feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as mM (row 3), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = xp.array([-1, 0, 1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = xp.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is QF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1))[0]
        ## Track which features are observed as MM (row 4), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = xp.array([0, 0, -1, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = xp.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is RF
        calc_idxs = xp.where((feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0))[0]
        ## Track which features are observed as MM (row 4), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = xp.array([0, -1, 0, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = xp.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = xp.append(all_used_inds[3], calc_idxs)
        #
    return all_use_cases, all_overlaps_options, all_used_inds


def identify_max_ESSs_vec(FF_inds, secondary_features, sorted_SGs_idxs, chunksize: Optional[int] = None):
    """
    Vectorized version of identify_max_ESSs that processes multiple fixed features at once.
    """
    sample_cardinality = global_scaled_matrix.shape[0]
    n_features = len(FF_inds)
    n_clusters = secondary_features.shape[1] - 1  # -1 because we delete the last cluster
    # Extract fixed features
    fixed_features = global_scaled_matrix[:, FF_inds]
    # Calculate fixed feature statistics
    if xpsparse.issparse(fixed_features):
        if USING_GPU:
            fixed_features_cardinality = fixed_features.sum(axis=0).flatten()
        else:
            fixed_features_cardinality = fixed_features.sum(axis=0).A.flatten()
    else:
        fixed_features_cardinality = fixed_features.sum(axis=0)
    fixed_feature_minority_states = fixed_features_cardinality.copy()
    idxs = fixed_features_cardinality >= (sample_cardinality / 2)
    if idxs.any():
        fixed_feature_minority_states[idxs] = sample_cardinality - fixed_features_cardinality[idxs]
    ## Get sort orders for all features (delete last cluster from each)
    all_sort_orders = xp.zeros((n_features, n_clusters), dtype=xp.int32)
    for i, FF_ind_global in enumerate(FF_inds):
        all_sort_orders[i] = xp.delete(sorted_SGs_idxs[FF_ind_global, :], -1)
    ## Calculate secondary feature sums and minority states for all features
    # Each feature has a unique sort order, so we need arrays for each
    all_SF_sums = xp.zeros((n_features, n_clusters), dtype=backend.dtype)
    all_SF_minority_states = xp.zeros((n_features, n_clusters), dtype=backend.dtype)
    all_FF_QF_vs_RF = xp.zeros((n_features, n_clusters), dtype=bool)
    # Loop over features to compute cumulative sums and related stats
    # NOTE: Little fiddly to vec due to unique sort orders per feature, not perf critical, loop OK for now
    for i in range(n_features):
        sort_order = all_sort_orders[i]
        if xpsparse.issparse(secondary_features):
            sf_reordered = secondary_features.A[:, sort_order]
        else:
            sf_reordered = secondary_features[:, sort_order]
        sf_cumsum = xp.cumsum(sf_reordered, axis=1)
        SF_sums = sf_cumsum.sum(axis=0)
        SF_minority_states = SF_sums.copy()
        SF_minority_states[SF_minority_states >= (sample_cardinality / 2)] = (
            sample_cardinality - SF_minority_states[SF_minority_states >= (sample_cardinality / 2)]
        )
        all_SF_sums[i] = SF_sums
        all_SF_minority_states[i] = SF_minority_states
        ## Identify where FF is the QF or RF
        FF_QF_vs_RF = xp.zeros(SF_minority_states.shape[0])
        FF_QF_vs_RF[xp.where(fixed_feature_minority_states[i] > SF_minority_states)[0]] = 1
        all_FF_QF_vs_RF[i] = FF_QF_vs_RF
    # Calculate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = xp.where(~all_FF_QF_vs_RF, fixed_feature_minority_states[:, None], all_SF_minority_states)
    QFms = xp.where(all_FF_QF_vs_RF, fixed_feature_minority_states[:, None], all_SF_minority_states)
    RFMs = sample_cardinality - RFms
    QFMs = sample_cardinality - QFms
    # Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = xp.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])
    # Calculate the overlap between the FF states and the secondary features
    overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup = (
        identify_max_ESSs_compute_overlaps_batch(
            fixed_features,
            fixed_features_cardinality,
            secondary_features,
            all_sort_orders,
            sample_cardinality,
            all_SF_sums,
            all_FF_QF_vs_RF,
        )
    )
    # Having extracted the overlaps, calculate the ESS and EPs using chunked computation
    all_ESSs, all_D_EPs, all_O_EPs, all_SWs, all_SGs = calc_ESSs_chunked(
        RFms,
        QFms,
        RFMs,
        QFMs,
        max_ent_options,
        sample_cardinality,
        overlaps,
        inverse_overlaps,
        case_idxs,
        case_patterns,
        overlap_lookup,
        xp_mod=xp,
        chunksize=chunksize,
    )
    # Postprocess EPs
    iden_feats, iden_cols = xp.nonzero(all_ESSs == 1)
    all_D_EPs[iden_feats, iden_cols] = 0
    all_O_EPs[iden_feats, iden_cols] = 0
    all_EPs = nanmaximum(all_D_EPs, all_O_EPs)
    return all_ESSs, all_EPs


def identify_max_ESSs_compute_overlaps_batch(
    fixed_features,
    fixed_features_cardinality,
    secondary_features,
    sort_orders,
    sample_cardinality,
    all_SF_sums,
    all_FF_QF_vs_RF,
):
    """
    Compute overlaps between multiple fixed features and their respective cumulative secondary features.
    Also computes case indices and lookup tables needed for ESS calculation.

    Each fixed feature has a unique sort_order that determines how secondary features are reordered
    and cumulatively summed.

    Parameters
    ----------
    fixed_features : sparse matrix (n_samples x n_fixed_features)
        The fixed features to compare
    secondary_features : sparse matrix (n_samples x n_clusters)
        The secondary features (one-hot clusters)
    sort_orders : array (n_fixed_features x n_clusters)
        Sort order for each fixed feature
    sample_cardinality : int
        Number of samples
    all_SF_sums : array (n_fixed_features x n_clusters)
        Secondary feature sums for each feature
    all_FF_QF_vs_RF : array (n_fixed_features x n_clusters)
        QF vs RF assignments for each feature

    Returns
    -------
    overlaps : array (n_fixed_features x n_clusters)
        Overlap values for each fixed feature with its cumulative secondary features
    inverse_overlaps : array (n_fixed_features x n_clusters)
        Inverse overlap values
    case_idxs : array (n_fixed_features x n_clusters)
        Case indices for each feature-cluster pair
    case_patterns : array (8 x 4)
        Case pattern lookup table
    overlap_lookup : array (8 x 4)
        Overlap source lookup table
    """
    n_fixed_features = fixed_features.shape[1] if fixed_features.shape[1] > 1 else len(sort_orders)
    n_clusters = sort_orders.shape[1]

    if USING_GPU:
        # Use CUDA kernel for GPU processing
        overlaps, inverse_overlaps = identify_max_ESSs_overlaps_cuda(
            fixed_features,
            secondary_features,
            sort_orders,
            sample_cardinality,
            n_fixed_features,
            n_clusters,
        )
    else:
        # Call numba-accelerated function for rare vectorised CPU use
        overlaps, inverse_overlaps = identify_max_ESSs_overlaps_numba(
            fixed_features.data.astype(np.float32),
            fixed_features.indices.astype(np.int32),
            fixed_features.indptr.astype(np.int32),
            secondary_features.data.astype(np.float32),
            secondary_features.indices.astype(np.int32),
            secondary_features.indptr.astype(np.int32),
            sort_orders.astype(np.int32).ravel(),
            sample_cardinality,
            n_fixed_features,
            n_clusters,
        )
        # Convert back to xp arrays if needed
        if xp != np:
            overlaps = xp.asarray(overlaps)
            inverse_overlaps = xp.asarray(inverse_overlaps)
    # Same case idx calculation as before
    ff_is_min = (fixed_features_cardinality[:, None] < (sample_cardinality / 2))
    sf_is_min = (all_SF_sums < (sample_cardinality / 2))
    case_idxs = (
        (ff_is_min.astype(int) << 2) + (sf_is_min.astype(int) << 1) + all_FF_QF_vs_RF.astype(int)
    ).astype(xp.int8)
    # Define case patterns (same for all features)
    case_patterns = xp.array(
        [
            [0, -1, 0, 1],  # case_8: ff=0, sf=0, FF_QF_vs_RF=0
            [0, 0, -1, 1],  # case_7: ff=0, sf=0, FF_QF_vs_RF=1
            [-1, 0, 1, 0],  # case_6: ff=0, sf=1, FF_QF_vs_RF=0
            [-1, 1, 0, 0],  # case_5: ff=0, sf=1, FF_QF_vs_RF=1
            [0, 1, 0, -1],  # case_4: ff=1, sf=0, FF_QF_vs_RF=0
            [0, 0, 1, -1],  # case_3: ff=1, sf=0, FF_QF_vs_RF=1
            [1, 0, -1, 0],  # case_2: ff=1, sf=1, FF_QF_vs_RF=0
            [1, -1, 0, 0],  # case_1: ff=1, sf=1, FF_QF_vs_RF=1
        ],
        dtype=xp.int8,
    )
    # Get overlap lookup
    row_map = xp.stack(
        [xp.argmax(case_patterns, axis=1), xp.argmin(case_patterns, axis=1)],
        axis=1,
        dtype=xp.int8,
    )
    overlap_lookup = xp.full((8, 4), -1, dtype=xp.int8)
    # Put_along cupy fix as before
    if USING_GPU:
        overlap_lookup = xp.asnumpy(overlap_lookup).astype(xp.int8)
        row_map = xp.asnumpy(row_map).astype(xp.int8)
        np.put_along_axis(overlap_lookup, row_map, [0, 1], axis=1)
        overlap_lookup = xp.asarray(overlap_lookup, dtype=xp.int8)
    else:
        xp.put_along_axis(overlap_lookup, row_map, [0, 1], axis=1)
    return overlaps, inverse_overlaps, case_idxs, case_patterns, overlap_lookup


@njit(parallel=True, cache=True)
def identify_max_ESSs_overlaps_numba(
    ff_data,
    ff_indices,
    ff_indptr,
    sf_data,
    sf_indices,
    sf_indptr,
    sort_orders,
    n_samples,
    n_fixed_features,
    n_clusters,
):
    """
    Numba-accelerated CPU version for computing overlaps with per-feature sort orders.

    Computes min(ff[i], cumsum(sf[:, sort_order[i]])[:, j]) for each i, j combination
    using sparse CSC format. CPU equivalent of identify_max_ESSs_overlaps_cuda.
    Parameters
    ----------
    ff_data, ff_indices, ff_indptr : arrays
        Fixed features in CSC sparse format
    sf_data, sf_indices, sf_indptr : arrays
        Secondary features in CSC sparse format
    sort_orders : array (n_fixed_features × n_clusters, flattened)
        Per-feature sort orders for reordering secondary features
    overlaps, inverse_overlaps : arrays (n_fixed_features × n_clusters)
        Output arrays (modified in-place)
    n_samples, n_fixed_features, n_clusters : int
        Matrix dimensions
    """
    overlaps = np.zeros((n_fixed_features, n_clusters), dtype=np.float32)
    inverse_overlaps = np.zeros((n_fixed_features, n_clusters), dtype=np.float32)
    # Parallel loop over fixed features and clusters
    for ff_idx in prange(n_fixed_features):
        for cumsum_idx in range(n_clusters):
            # Get fixed feature column range
            ff_start = ff_indptr[ff_idx]
            ff_end = ff_indptr[ff_idx + 1]
            overlap = 0.0
            inverse_overlap = 0.0
            # For each sample (row)
            for sample in range(n_samples):
                # Get fixed feature value at this sample
                ff_val = 0.0
                for ptr in range(ff_start, ff_end):
                    if ff_indices[ptr] == sample:
                        ff_val = ff_data[ptr]
                        break
                # Compute cumulative sum of secondary features up to cumsum_idx
                # using this fixed feature's sort order
                sf_cumsum = 0.0
                for k in range(cumsum_idx + 1):
                    # Get the cluster index from sort_order
                    cluster_idx = sort_orders[ff_idx * n_clusters + k]
                    # Get secondary feature value at this sample for this cluster
                    sf_start = sf_indptr[cluster_idx]
                    sf_end = sf_indptr[cluster_idx + 1]
                    for ptr in range(sf_start, sf_end):
                        if sf_indices[ptr] == sample:
                            sf_cumsum += sf_data[ptr]
                            break
                # Accumulate min(ff, sf_cumsum) for overlap
                overlap += min(ff_val, sf_cumsum)
                # Accumulate min(1-ff, sf_cumsum) for inverse overlap
                inverse_overlap += min(1.0 - ff_val, sf_cumsum)
            overlaps[ff_idx, cumsum_idx] = overlap
            inverse_overlaps[ff_idx, cumsum_idx] = inverse_overlap
    return overlaps, inverse_overlaps


def identify_max_ESSs_overlaps_cuda(
    fixed_features,
    secondary_features,
    sort_orders,
    sample_cardinality,
    n_fixed_features,
    n_clusters,
):
    """
    CUDA kernel for computing overlaps with per-feature sort orders and cumulative sums.
    This kernel computes min(ff[i], cumsum(sf[:, sort_order[i]])[:, j]) for each i, j combination.
    """
    kernel_code = r"""
    extern "C" __global__
    void compute_max_esss_overlaps(
        const float* ff_data,
        const int* ff_indices,
        const int* ff_indptr,
        const float* sf_data,
        const int* sf_indices,
        const int* sf_indptr,
        const int* sort_orders,
        float* overlaps,
        float* inverse_overlaps,
        int n_samples,
        int n_fixed_features,
        int n_clusters
    ) {
        int ff_idx = blockIdx.x * blockDim.x + threadIdx.x;  // fixed feature index
        int cumsum_idx = blockIdx.y * blockDim.y + threadIdx.y;  // cumulative cluster index

        if (ff_idx >= n_fixed_features || cumsum_idx >= n_clusters) return;

        // Get fixed feature column
        int ff_start = ff_indptr[ff_idx];
        int ff_end = ff_indptr[ff_idx + 1];

        float overlap = 0.0f;
        float inverse_overlap = 0.0f;

        // For each sample (row)
        for (int sample = 0; sample < n_samples; sample++) {
            // Get fixed feature value at this sample
            float ff_val = 0.0f;
            for (int ptr = ff_start; ptr < ff_end; ptr++) {
                if (ff_indices[ptr] == sample) {
                    ff_val = ff_data[ptr];
                    break;
                }
            }

            // Compute cumulative sum of secondary features up to cumsum_idx
            // using this fixed feature's sort order
            float sf_cumsum = 0.0f;
            for (int k = 0; k <= cumsum_idx; k++) {
                // Get the cluster index from sort_order
                int cluster_idx = sort_orders[ff_idx * n_clusters + k];

                // Get secondary feature value at this sample for this cluster
                int sf_start = sf_indptr[cluster_idx];
                int sf_end = sf_indptr[cluster_idx + 1];

                for (int ptr = sf_start; ptr < sf_end; ptr++) {
                    if (sf_indices[ptr] == sample) {
                        sf_cumsum += sf_data[ptr];
                        break;
                    }
                }
            }

            // Accumulate min(ff, sf_cumsum) for overlap
            overlap += fminf(ff_val, sf_cumsum);

            // Accumulate min(1-ff, sf_cumsum) for inverse overlap
            // Assuming ff_val is in [0, 1] range after scaling
            inverse_overlap += fminf(1.0f - ff_val, sf_cumsum);
        }

        overlaps[ff_idx * n_clusters + cumsum_idx] = overlap;
        inverse_overlaps[ff_idx * n_clusters + cumsum_idx] = inverse_overlap;
    }
    """
    module = xp.RawModule(code=kernel_code)
    kernel = module.get_function("compute_max_esss_overlaps")
    # Prepare arrays
    overlaps = xp.zeros((n_fixed_features, n_clusters), dtype=xp.float32)
    inverse_overlaps = xp.zeros((n_fixed_features, n_clusters), dtype=xp.float32)
    # Convert to CSC format and ensure contiguous
    ff_csc = fixed_features.tocsc()
    sf_csc = secondary_features.tocsc()
    # Launch kernel
    block = (16, 16)
    grid = (
        (n_fixed_features + block[0] - 1) // block[0],
        (n_clusters + block[1] - 1) // block[1],
    )
    kernel(
        grid,
        block,
        (
            ff_csc.data.astype(xp.float32),
            ff_csc.indices,
            ff_csc.indptr,
            sf_csc.data.astype(xp.float32),
            sf_csc.indices,
            sf_csc.indptr,
            sort_orders.astype(xp.int32).ravel(),
            overlaps.ravel(),
            inverse_overlaps.ravel(),
            sample_cardinality,
            n_fixed_features,
            n_clusters,
        ),
    )
    return overlaps, inverse_overlaps

##### Find minimal set of marker genes functions #####


def ES_FMG(
    adata,
    N,
    secondary_features_label,
    input_genes: Optional[tuple[str]] = None,
    num_reheats: int = 3,
    resolution: int = 1,
    use_cores: int = -1,
):
    """
    Having used ES_CCF identify a set of features/clusters that maximise the ESS of each variable/column in adata and parallel_calc_es_matrices
    to calculate the ESSs of every varible/column in adata in relation to each ESS_Max feature/cluster, we can now use ES_FMG
    to identify a set of N clusters that maximally capture distinct gene expression patterns in the counts matrix of adata.
    """
    #
    cores_avail = multiprocess.cpu_count()
    print("Cores Available: " + str(cores_avail))
    if use_cores == -1:
        use_cores = (
            cores_avail - 1
        )  # -1 Is an arbitrary buffer of idle cores that I set.
        if use_cores < 1:
            use_cores = 1
    print("Cores Used: " + str(use_cores))
    # Gene set optimisation
    if input_genes is None:
        # input_genes = xp.array(adata.var_names.tolist())
        input_genes = adata.var_names.tolist()
        # No need to calc, since the input genes are from our adata
        input_gene_idxs = range(len(input_genes))
    else:
        input_gene_idxs = adata.var_names.get_indexer(input_genes)
    #
    ###
    global clust_ESSs
    clust_ESSs = xp.asarray(adata.varm[secondary_features_label + "_ESSs"])[
        xp.ix_(input_gene_idxs, input_gene_idxs)
    ]
    clust_ESSs[clust_ESSs < 0] = 0
    ###
    max_ESSs = xp.max(clust_ESSs, axis=1)
    ###
    # TODO: Control seed for reproducibility?
    chosen_clusts = xp.random.choice(clust_ESSs.shape[0], N, replace=False)
    ###
    best_score = -xp.inf
    reheat = 1
    while reheat <= num_reheats:
        ###
        chosen_pairwise_ESSs = clust_ESSs[xp.ix_(chosen_clusts, chosen_clusts)]
        xp.fill_diagonal(chosen_pairwise_ESSs, 0)
        current_score = xp.sum(
            max_ESSs[chosen_clusts]
            - (xp.max(chosen_pairwise_ESSs, axis=1) * resolution)
        )
        ###
        end = 0
        ###
        while end == 0:
            ###
            all_max_changes, all_max_change_idxs, all_max_replacement_scores = (
                parallel_replace_clust(
                    N,
                    chosen_pairwise_ESSs,
                    chosen_clusts,
                    current_score,
                    max_ESSs,
                    resolution,
                    use_cores=use_cores,
                )
            )
            ###
            replace_clust_idx = xp.argmax(all_max_changes)
            if all_max_changes[replace_clust_idx] > 0:
                # print(all_max_replacement_scores[replace_clust_idx])
                replacement_clust_idx = all_max_change_idxs[replace_clust_idx]
                #
                # print(Replacement_Gene_Ind)
                chosen_clusts[replace_clust_idx] = replacement_clust_idx
                #
                chosen_pairwise_ESSs = clust_ESSs[xp.ix_(chosen_clusts, chosen_clusts)]
                xp.fill_diagonal(chosen_pairwise_ESSs, 0)
                current_score = xp.sum(
                    max_ESSs[chosen_clusts]
                    - (xp.max(chosen_pairwise_ESSs, axis=1) * resolution)
                )
                #
                # print(current_score)
                if current_score > best_score:
                    best_score = current_score
                    print("Current highest score: " + str(best_score))
                    best_chosen_clusters = chosen_clusts.copy()
            else:
                end = 1
        #
        if reheat <= num_reheats:
            reheat_num = int(np.ceil(N * 0.25))
            random_reheat_1 = np.random.choice(
                input_genes.shape[0], reheat_num, replace=False
            )
            random_reheat_2 = np.random.randint(chosen_clusts.shape[0], size=reheat_num)
            chosen_clusts[random_reheat_2] = random_reheat_1
            print(f"Reheat number: {reheat}")
            reheat += 1
    #
    chosen_pairwise_ESSs = clust_ESSs[xp.ix_(chosen_clusts, chosen_clusts)]
    return (
        best_chosen_clusters,
        [input_genes[i] for i in best_chosen_clusters],
        chosen_pairwise_ESSs,
    )


def replace_clust(
    replace_idx,
    chosen_pairwise_ESSs,
    chosen_clusts,
    current_score,
    max_ESSs,
    resolution,
):
    sub_chosen_clusts = xp.delete(chosen_clusts, replace_idx)
    #
    sub_chosen_pairwise_ESSs = xp.delete(
        chosen_pairwise_ESSs, replace_idx, axis=0
    )  # Delete a row, which should be the cluster
    sub_chosen_pairwise_ESSs = xp.delete(sub_chosen_pairwise_ESSs, replace_idx, axis=1)
    sub_2nd_maxs = xp.max(sub_chosen_pairwise_ESSs, axis=1)
    replacement_columns = clust_ESSs[sub_chosen_clusts, :]
    replacement_columns[(xp.arange(sub_chosen_clusts.shape[0]), sub_chosen_clusts)] = 0
    sub_2nd_maxs = xp.maximum(sub_2nd_maxs[:, xp.newaxis], replacement_columns)
    #
    replacement_scores_1 = xp.sum(
        (max_ESSs[sub_chosen_clusts, xp.newaxis] - (sub_2nd_maxs * resolution)), axis=0
    )
    #
    replacement_rows = clust_ESSs[:, sub_chosen_clusts]
    replacement_rows[(sub_chosen_clusts, xp.arange(sub_chosen_clusts.shape[0]))] = 0
    #
    replacement_scores_2 = max_ESSs - (xp.max(replacement_rows, axis=1) * resolution)
    #
    replacement_scores = replacement_scores_1 + replacement_scores_2
    ###
    changes = replacement_scores - current_score
    changes[chosen_clusts] = -xp.inf
    max_idx = xp.argmax(changes)
    max_change = changes[max_idx]
    # all_max_changes[i] = max_change
    # all_max_change_idxs[i] = max_idx
    # all_max_replacement_scores[i] = replacement_scores[max_idx]
    #
    return max_change, max_idx, replacement_scores[max_idx]


def parallel_replace_clust(
    N,
    chosen_pairwise_ESSs,
    chosen_clusts,
    current_score,
    max_ESSs,
    resolution,
    use_cores,
):
    replace_idxs = xp.arange(N)
    #
    pool = multiprocess.Pool(processes=use_cores)
    results = pool.map(
        partial(
            replace_clust,
            resolution=resolution,
            chosen_pairwise_ESSs=chosen_pairwise_ESSs,
            chosen_clusts=chosen_clusts,
            current_score=current_score,
            max_ESSs=max_ESSs,
        ),
        replace_idxs,
    )
    pool.close()
    pool.join()
    results = xp.asarray(results)
    all_max_changes = results[:, 0]
    all_max_change_idxs = results[:, 1]
    all_max_replacement_scores = results[:, 2]
    return all_max_changes, all_max_change_idxs, all_max_replacement_scores
