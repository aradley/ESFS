###### ESFS ######

### Dependencies ###
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import issparse
import umap
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import warnings

from functools import partial
import multiprocess
from p_tqdm import p_map

import matplotlib.animation as animation
from IPython.display import Image, display
import tempfile
import os

from scipy.sparse import csc_matrix
from scipy.spatial.distance import pdist, squareform

### Dependencies ###


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
    keep_genes = keep_genes = adata.var_names[np.where(adata.X.getnnz(axis=0) > 50)[0]]
    if keep_genes.shape[0] < adata.shape[1]:
        print(
            str(adata.shape[1] - keep_genes.shape[0])
            + " genes show no expression. Removing them from adata object"
        )
        adata = adata[:, keep_genes]
    # Un-sparsify the data for clipping and scaling
    scaled_expressions = adata.X.copy()
    if issparse(scaled_expressions) == True:
        scaled_expressions = np.asarray(scaled_expressions.todense())
    # Log scale the data if user requests.
    if log_scale == True:
        scaled_expressions = np.log2(scaled_expressions + 1)
    # Clip exceptionally high gene expression for each gene. Default percentile is the 97.5th.
    upper = np.percentile(scaled_expressions, clip_percentile, axis=0)
    upper[np.where(upper == 0)[0]] = np.max(scaled_expressions, axis=0)[
        np.where(upper == 0)[0]
    ]
    scaled_expressions = scaled_expressions.clip(max=upper[None, :])
    # Normalise gene expression between 0 and 1.
    scaled_expressions = scaled_expressions / upper
    # Return data as a sparse csc_matrix
    adata.layers["Scaled_Counts"] = csc_matrix(scaled_expressions.astype("f"))
    print(
        "Scaled expression matrix has been saved to 'adata.layers['Scaled_Counts']' as a sparse csc_matrix"
    )
    return adata


def parallel_calc_es_matrices(
    adata,
    secondary_features_label="Self",
    save_matrices=np.array(["ESSs", "EPs"]),
    use_cores=-1,
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
    """
    ## Establish which secondary_features will be compared against each of the features in adata
    global secondary_features
    if secondary_features_label == "Self":
        secondary_features = adata.layers["Scaled_Counts"].copy()
    else:
        print(
            "You have provided a 'secondary_features_label', implying that in the anndata object there is a corresponding csc_sparse martix object with rows as samples and columns as features. Each feature will be used to calculate ES scores for each of the variables of the adata object"
        )
        secondary_features = adata.obsm[secondary_features_label]
    #
    ## Create the global global_scaled_matrix array for faster parallel computing calculations
    global global_scaled_matrix
    global_scaled_matrix = adata.layers["Scaled_Counts"]
    ## Extract sample and feature cardinality
    sample_cardinality = global_scaled_matrix.shape[0]
    ## Calculate feature sums and minority states for each adata feature
    feature_sums = global_scaled_matrix.sum(axis=0).A[0]
    minority_states = feature_sums.copy()
    idxs = np.where(minority_states >= (sample_cardinality / 2))[0]
    minority_states[idxs] = sample_cardinality - minority_states[idxs]
    ####
    ## Provide indicies for parallel computing.
    feature_inds = np.arange(secondary_features.shape[1])
    ## Identify number of cores to use.
    cores_avail = multiprocess.cpu_count()
    print("Cores Available: " + str(cores_avail))
    if use_cores == -1:
        use_cores = (
            cores_avail - 1
        )  # -1 Is an arbitrary buffer of idle cores that I set.
        if use_cores < 1:
            use_cores = 1
    print("Cores Used: " + str(use_cores))
    ## Perform calculations
    print("Calculating ESS and EP matricies.")
    print(
        "If progress bar freezes consider increasing system memory or reducing number of cores used with the 'use_cores' parameter as you may have hit a memory ceiling for your machine."
    )
    ## Parallel compute
    with np.errstate(divide="ignore", invalid="ignore"):
        if use_cores == 1:
            # Single core: run sequentially for easier debugging
            Results = [
                calc_es_metrics(
                    ind,
                    sample_cardinality=sample_cardinality,
                    feature_sums=feature_sums,
                    minority_states=minority_states,
                )
                for ind in feature_inds
            ]
        else:
            # Multi-core: use parallel processing
            Results = p_map(
                partial(
                    calc_es_metrics,
                    sample_cardinality=sample_cardinality,
                    feature_sums=feature_sums,
                    minority_states=minority_states,
                ),
                feature_inds,
                num_cpus=use_cores,
            )
    ## Unpack results
    Results = np.asarray(Results)
    ## Save outputs requested by the save_matrices paramater
    if np.isin("ESSs", save_matrices):
        ESSs = Results[:, 0, :]
        if (
            secondary_features_label == "Self"
        ):  ## The vast majority of outputs are symmetric, but float errors appear to make some non-symmetric. If we can fix this that could be cool.
            ESSs = nanmaximum(ESSs, ESSs.T)
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
    if np.isin("EPs", save_matrices):
        EPs = Results[:, 1, :]
        if secondary_features_label == "Self":
            EPs = nanmaximum(EPs, EPs.T)
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
    if np.isin("SWs", save_matrices):
        SWs = Results[:, 2, :]
        if secondary_features_label == "Self":
            SWs = nanmaximum(SWs, SWs.T)
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
    if np.isin("SGs", save_matrices):
        SGs = Results[:, 3, :]
        if secondary_features_label == "Self":
            SGs = nanmaximum(SGs, SGs.T)
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
        #
    return adata


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
    arr1[np.isinf(arr1)] = -np.inf
    arr2[np.isinf(arr2)] = -np.inf
    # Replace NaN values with -infinity for comparison
    arr1_nan = np.isnan(arr1)
    arr2_nan = np.isnan(arr2)
    arr1[arr1_nan] = -np.inf
    arr2[arr2_nan] = -np.inf
    # Compute the element-wise maximum
    result = np.maximum(arr1, arr2)
    # Where both values are NaN, the result should be NaN
    nan_mask = arr1_nan & arr2_nan
    result[nan_mask] = np.nan
    return result


def calc_es_metrics(feature_ind, sample_cardinality, feature_sums, minority_states):
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
    fixed_feature_cardinality = np.sum(fixed_feature)
    fixed_feature_minority_state = fixed_feature_cardinality
    if fixed_feature_minority_state >= (sample_cardinality / 2):
        fixed_feature_minority_state = sample_cardinality - fixed_feature_minority_state
    ## Identify where FF is the QF or RF
    FF_QF_vs_RF = np.zeros(feature_sums.shape[0])
    FF_QF_vs_RF[np.where(fixed_feature_minority_state > minority_states)[0]] = (
        1  # 1's mean FF is QF
    )
    ## Caclulate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = minority_states.copy()
    idxs = np.where(FF_QF_vs_RF == 0)[0]
    RFms[idxs] = fixed_feature_minority_state
    RFMs = sample_cardinality - RFms
    QFms = minority_states.copy()
    idxs = np.where(FF_QF_vs_RF == 1)[0]
    QFms[idxs] = fixed_feature_minority_state
    QFMs = sample_cardinality - QFms
    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = np.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])
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
    ESSs, D_EPs, O_EPs, SWs, SGs = calc_ESSs(
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
    identical_features = np.where(ESSs == 1)[0]
    D_EPs[identical_features] = 0
    O_EPs[identical_features] = 0
    EPs = nanmaximum(D_EPs, O_EPs)
    return ESSs, EPs, SWs, SGs


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
    all_use_cases = np.zeros((4, feature_sums.shape[0]))
    ## Set up an array to track the observed overlaps between the FF and the secondary features.
    all_overlaps_options = np.zeros((4, feature_sums.shape[0]))
    ## Set up a list to track the used inds/features for each ESE
    all_used_inds = [[]] * 4
    ####
    ## Pairwise calculate total overlaps of FF values with the values every other a feature in adata
    nonzero_inds = np.where(fixed_feature != 0)[0]
    sub_global_scaled_matrix = global_scaled_matrix[nonzero_inds, :]
    B = csc_matrix(
        (
            fixed_feature[nonzero_inds].T[0][sub_global_scaled_matrix.indices],
            sub_global_scaled_matrix.indices,
            sub_global_scaled_matrix.indptr,
        )
    )
    overlaps = sub_global_scaled_matrix.minimum(B).sum(axis=0).A[0]
    ## Pairwise calculate total overlaps of Inverse FF values with the values every other a feature in adata
    inverse_fixed_feature = 1 - fixed_feature  # np.max(fixed_feature) - fixed_feature
    nonzero_inds = np.where(inverse_fixed_feature != 0)[0]
    sub_global_scaled_matrix = global_scaled_matrix[nonzero_inds, :]
    B = csc_matrix(
        (
            inverse_fixed_feature[nonzero_inds].T[0][sub_global_scaled_matrix.indices],
            sub_global_scaled_matrix.indices,
            sub_global_scaled_matrix.indptr,
        )
    )
    inverse_overlaps = sub_global_scaled_matrix.minimum(B).sum(axis=0).A[0]
    ####
    ### Using the logical rules of ES to work out which ESE should be used for each pair of features beign compared.
    ## If FF is observed in it's minority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality < (sample_cardinality / 2):
        #######
        ## FF and other feature are minority states & FF is QF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as mm (row 1), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = np.array([1, -1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        #######
        ## FF and other feature are minority states & FF is RF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as mm (row 1), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = np.array([1, 0, -1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is QF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as mM (row 4), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = np.array([0, 0, 1, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is RF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as Mm (row 2), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = np.array([0, 1, 0, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        #
    ## If FF is observed in it's majority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality >= (sample_cardinality / 2):
        #######
        ## FF is majority, other feature is minority & FF is QF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as Mm (row 2), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = np.array([-1, 1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is minority & FF is RF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as mM (row 3), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = np.array([-1, 0, 1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is QF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as MM (row 4), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = np.array([0, 0, -1, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is RF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as MM (row 4), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = np.array([0, -1, 0, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        #
    return all_use_cases, all_overlaps_options, all_used_inds


def calc_ESSs(
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
    all_ESSs = np.zeros((4, RFms.shape[0]))
    all_D_EPs = np.zeros((4, RFms.shape[0]))
    all_O_EPs = np.zeros((4, RFms.shape[0]))
    all_SGs = np.zeros((4, RFms.shape[0]))
    all_SWs = np.zeros((4, RFms.shape[0]))
    ###################
    ##### (1)  mm #####
    use_curve = 0
    ## Find the FF/SF pairs where we should use ESE (1) to calculate entropies
    calc_idxs = all_used_inds[use_curve].astype("i")
    if calc_idxs.shape[0] > 0:
        # Retrieve the max_ent, Min_x, Max_X and observed overlap values
        min_overlap = np.repeat(0, calc_idxs.shape[0])
        max_overlap = RFms[calc_idxs]
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = np.where(overlaps < max_ent_x)[0]
        SD1_idxs = np.where(overlaps >= max_ent_x)[0]
        SDs = np.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = np.zeros(calc_idxs.shape[0])
        O = np.zeros(calc_idxs.shape[0])
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
        D_EPs = np.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = (
            (CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]
        ) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = np.zeros(ind_E.shape[0])
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
        min_overlap = np.repeat(0, calc_idxs.shape[0])
        max_overlap = np.minimum(RFms[calc_idxs], QFMs[calc_idxs])
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = np.where(overlaps < max_ent_x)[0]
        SD1_idxs = np.where(overlaps >= max_ent_x)[0]
        SDs = np.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = np.zeros(calc_idxs.shape[0])
        O = np.zeros(calc_idxs.shape[0])
        D[SD_1_idxs] = overlaps[SD_1_idxs]
        O[SD_1_idxs] = (
            QFms[calc_idxs][SD_1_idxs]
            - RFms[calc_idxs][SD_1_idxs]
            + overlaps[SD_1_idxs]
        )
        D[SD1_idxs] = RFms[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        O[SD1_idxs] = (
            QFMs[calc_idxs][SD1_idxs] - RFms[calc_idxs][SD1_idxs] + D[SD1_idxs]
        )
        # Perform caclulations with ESE (2)
        CE, ind_E, min_E = ESE2(
            overlaps,
            SDs,
            RFms[calc_idxs],
            RFMs[calc_idxs],
            QFms[calc_idxs],
            QFMs[calc_idxs],
            sample_cardinality,
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
        D_EPs = np.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = (
            (CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]
        ) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = np.zeros(ind_E.shape[0])
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
        max_overlap = np.minimum(QFms[calc_idxs], RFMs[calc_idxs])
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = np.where(overlaps < max_ent_x)[0]
        SD1_idxs = np.where(overlaps >= max_ent_x)[0]
        SDs = np.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = np.zeros(calc_idxs.shape[0])
        O = np.zeros(calc_idxs.shape[0])
        D[SD_1_idxs] = (
            QFMs[calc_idxs][SD_1_idxs]
            - RFMs[calc_idxs][SD_1_idxs]
            + overlaps[SD_1_idxs]
        )
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
        D_EPs = np.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = (
            (CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]
        ) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = np.zeros(ind_E.shape[0])
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
        max_overlap = np.minimum(QFMs[calc_idxs], RFMs[calc_idxs])
        overlaps = all_overlaps_options[use_curve, calc_idxs]
        max_ent_x = max_ent_options[use_curve, calc_idxs]
        ind_X_1 = max_ent_x - min_overlap
        ind_X1 = max_overlap - max_ent_x
        #
        SD_1_idxs = np.where(overlaps < max_ent_x)[0]
        SD1_idxs = np.where(overlaps >= max_ent_x)[0]
        SDs = np.zeros(calc_idxs.shape[0]) - 1
        SDs[SD1_idxs] = 1
        #
        D = np.zeros(calc_idxs.shape[0])
        O = np.zeros(calc_idxs.shape[0])
        D[SD_1_idxs] = overlaps[SD_1_idxs] - (
            sample_cardinality
            - (QFms[calc_idxs][SD_1_idxs] + RFms[calc_idxs][SD_1_idxs])
        )
        O[SD_1_idxs] = overlaps[SD_1_idxs]
        D[SD1_idxs] = QFMs[calc_idxs][SD1_idxs] - overlaps[SD1_idxs]
        O[SD1_idxs] = (
            RFMs[calc_idxs][SD1_idxs] - QFMs[calc_idxs][SD1_idxs] + D[SD1_idxs]
        )
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
        D_EPs = np.zeros(ind_E.shape[0])
        D_EPs[SD_1_idxs] = (
            (CE[SD_1_idxs] - min_E[SD_1_idxs]) / D[SD_1_idxs]
        ) - SD_1_IndEnt
        D_EPs[SD1_idxs] = ((CE[SD1_idxs] - min_E[SD1_idxs]) / D[SD1_idxs]) - SD1_IndEnt
        #
        O_EPs = np.zeros(ind_E.shape[0])
        O_EPs[SD_1_idxs] = ((CE[SD_1_idxs]) / O[SD_1_idxs]) - SD_1_IndEnt
        O_EPs[SD1_idxs] = ((CE[SD1_idxs]) / O[SD1_idxs]) - SD1_IndEnt
        #
        all_D_EPs[use_curve, calc_idxs] = D_EPs
        all_O_EPs[use_curve, calc_idxs] = O_EPs
        #
    ########
    ## For each feature pair, accept the orientation with the maximum ESS as it is the least likely to have occoured by chance.
    max_ESS_idxs = np.nanargmax(np.absolute(all_ESSs), axis=0)
    ## Return results
    return (
        all_ESSs[max_ESS_idxs, np.arange(RFms.shape[0])],
        all_D_EPs[max_ESS_idxs, np.arange(RFms.shape[0])],
        all_O_EPs[max_ESS_idxs, np.arange(RFms.shape[0])],
        all_SWs[max_ESS_idxs, np.arange(RFms.shape[0])],
        all_SGs[max_ESS_idxs, np.arange(RFms.shape[0])],
    )


def ESE1(x, SD, RFm, RFM, QFm, QFM, Ts, max_overlap):
    """
    This function takes the observed inputs and uses the ESE1 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (((x) / RFm) * (-np.log((x) / RFm)))
        + (((RFm - x) / RFm) * (-np.log((RFm - x) / RFm)))
    )
    G2_E = (RFM / Ts) * (
        (((QFm - x) / RFM) * (-np.log((QFm - x) / RFM)))
        + (((RFM - QFm + x) / RFM) * (-np.log((RFM - QFm + x) / RFM)))
    )
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-np.log((QFm / Ts))) + (QFM / Ts) * (-np.log((QFM / Ts)))
    #
    min_E = np.zeros(SD.shape[0])
    SD_1_idxs = np.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (
            ((QFm[SD_1_idxs]) / RFM[SD_1_idxs])
            * (-np.log((QFm[SD_1_idxs]) / RFM[SD_1_idxs]))
        )
        + (
            ((RFM[SD_1_idxs] - QFm[SD_1_idxs]) / RFM[SD_1_idxs])
            * (-np.log((RFM[SD_1_idxs] - QFm[SD_1_idxs]) / RFM[SD_1_idxs]))
        )
    )
    SD1_idxs = np.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            ((QFm[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            * (-np.log((QFm[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs]))
        )
        + (
            ((RFM[SD1_idxs] - QFm[SD1_idxs] + max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            * (
                -np.log(
                    (RFM[SD1_idxs] - QFm[SD1_idxs] + max_overlap[SD1_idxs])
                    / RFM[SD1_idxs]
                )
            )
        )
    )
    min_E[np.isnan(min_E)] = 0
    min_E[np.isnan(min_E)] = 0
    #
    CE[np.isnan(CE)] = min_E[np.isnan(CE)]
    return CE, ind_E, min_E


def ESE2(x, SD, RFm, RFM, QFm, QFM, Ts):
    """
    This function takes the observed inputs and uses the ESE2 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (
            -(((RFm - x) / RFm) * np.log((RFm - x) / RFm))
            - (((x) / RFm) * np.log((x) / RFm))
        )
    )
    G2_E = (RFM / Ts) * (
        (
            -(((RFM - QFM + x) / RFM) * np.log((RFM - QFM + x) / RFM))
            - (((QFM - x) / RFM) * np.log((QFM - x) / RFM))
        )
    )
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-np.log((QFm / Ts))) + (QFM / Ts) * (-np.log((QFM / Ts)))
    #
    min_E = np.zeros(SD.shape[0])
    SD_1_idxs = np.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD_1_idxs] - QFM[SD_1_idxs]) / RFM[SD_1_idxs])
                * np.log((RFM[SD_1_idxs] - QFM[SD_1_idxs]) / RFM[SD_1_idxs])
            )
            - (
                ((QFM[SD_1_idxs]) / RFM[SD_1_idxs])
                * np.log((QFM[SD_1_idxs]) / RFM[SD_1_idxs])
            )
        )
    )
    SD1_idxs = np.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs])
                * np.log(
                    (RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs]
                )
            )
            - (
                ((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
                * np.log((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
            )
        )
    )
    min_E[np.isnan(min_E)] = 0
    #
    CE[np.isnan(CE)] = min_E[np.isnan(CE)]
    return CE, ind_E, min_E


def ESE3(x, SD, RFm, RFM, QFm, QFM, Ts, min_overlap):
    """
    This function takes the observed inputs and uses the ESE3 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (
            -(((QFm - x) / RFm) * np.log((QFm - x) / RFm))
            - (((RFm - QFm + x) / RFm) * np.log((RFm - QFm + x) / RFm))
        )
    )
    G2_E = (RFM / Ts) * (
        (
            -(((x) / RFM) * np.log((x) / RFM))
            - (((RFM - x) / RFM) * np.log((RFM - x) / RFM))
        )
    )
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-np.log((QFm / Ts))) + (QFM / Ts) * (-np.log((QFM / Ts)))
    #
    min_E = np.zeros(SD.shape[0])
    SD_1_idxs = np.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (
            -(
                ((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * np.log((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
            - (
                ((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * np.log((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
        )
    )
    SD1_idxs = np.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs])
                * np.log(
                    (RFM[SD1_idxs] - QFM[SD1_idxs] + RFm[SD1_idxs]) / RFM[SD1_idxs]
                )
            )
            - (
                ((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
                * np.log((QFM[SD1_idxs] - RFm[SD1_idxs]) / RFM[SD1_idxs])
            )
        )
    )
    min_E[np.isnan(min_E)] = 0
    #
    CE[np.isnan(CE)] = min_E[np.isnan(CE)]
    return CE, ind_E, min_E


def ESE4(x, SD, RFm, RFM, QFm, QFM, Ts, min_overlap, max_overlap):
    """
    This function takes the observed inputs and uses the ESE4 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (ind_E) and Minimum Entropy (min_E).
    """
    G1_E = (RFm / Ts) * (
        (
            -(((RFm - QFM + x) / RFm) * np.log((RFm - QFM + x) / RFm))
            - (((QFM - x) / RFm) * np.log((QFM - x) / RFm))
        )
    )
    G2_E = (RFM / Ts) * (
        (
            -(((RFM - x) / RFM) * np.log((RFM - x) / RFM))
            - (((x) / RFM) * np.log((x) / RFM))
        )
    )
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    ind_E = (QFm / Ts) * (-np.log((QFm / Ts))) + (QFM / Ts) * (-np.log((QFM / Ts)))
    #
    min_E = np.zeros(SD.shape[0])
    SD_1_idxs = np.where(SD == -1)[0]
    min_E[SD_1_idxs] = (RFM[SD_1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * np.log((RFM[SD_1_idxs] - min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
            - (
                ((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
                * np.log((min_overlap[SD_1_idxs]) / RFM[SD_1_idxs])
            )
        )
    )
    SD1_idxs = np.where(SD == 1)[0]
    min_E[SD1_idxs] = (RFM[SD1_idxs] / Ts) * (
        (
            -(
                ((RFM[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs])
                * np.log((RFM[SD1_idxs] - max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            )
            - (
                ((max_overlap[SD1_idxs]) / RFM[SD1_idxs])
                * np.log((max_overlap[SD1_idxs]) / RFM[SD1_idxs])
            )
        )
    )
    min_E[np.isnan(min_E)] = 0
    #
    CE[np.isnan(CE)] = min_E[np.isnan(CE)]
    return CE, ind_E, min_E


##### Combinatorial cluster marker gene identification functions #####

## For any dataset, we may cluster the samples into groups. We may then be interested in which features best describe different groupings
# of the data. However, we may not know what resolution of the data best describes the patterns we are interested in. This ambuguity may
# be remedied by looking at every combination of clusters for an intentionally overclustered dataset. However, searching every combination
# of clusters quickly becomes computationally intractible.  To overcome this challange, we introduce a combinatorial clustering algorithm
# that turns the the combinatorial problem into a linear one, which can be tractably solved in mamy practical scenarios.


def find_max_ESSs(adata, secondary_features_label):
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
    print(
        "For this function to work, you must have ran the parallel_calc_es_matrices function in a manner that attaches ESS and SG objects to your adata object that relate to your secondary_features_label label."
    )
    ## Create the global global_scaled_matrix array for faster parallel computing calculations
    global global_scaled_matrix
    global_scaled_matrix = adata.layers["Scaled_Counts"]
    ### Extract the secondary_features object from adata
    secondary_features = adata.obsm[secondary_features_label]
    ### Extract the secondary_features ESSs from adata
    all_ESSs = adata.varm[secondary_features_label + "_ESSs"]
    initial_max_ESSs = np.asarray(np.max(all_ESSs, axis=1))
    max_ESSs = initial_max_ESSs.copy()
    ### Extract the secondary_features SGs from adata
    all_SGs = np.asarray(adata.varm[secondary_features_label + "_SGs"]).copy()
    all_SGs[all_ESSs < 0] = (
        all_SGs[all_ESSs < 0] * -1
    )  # For ordering we need to include the SG sort directions which we can extract from the ESSs
    ### For each feature in adata, sort the SGs of the secondary_features from highest to lowest. This is the main step that allows
    # use to turn the intractible combinatorial problem into a tractable linear problem.
    sorted_SGs_idxs = np.argsort(all_SGs)[:, ::-1]
    ### Parallel warpper function
    max_ESSs, max_EPs, top_score_columns_combinations, top_score_secondary_features = (
        parallel_identify_max_ESSs(secondary_features, sorted_SGs_idxs, use_cores=-1)
    )
    ## Compile results
    combinatorial_label_info = np.column_stack(
        [np.array(max_ESSs), np.array(max_EPs)]
    )  # ,np.array(top_score_columns_combinations,dtype="object")])
    combinatorial_label_info = pd.DataFrame(
        combinatorial_label_info, index=adata.var_names, columns=["Max_ESSs", "EPs"]
    )  # ,"top_score_columns_combinations"])
    ### Save Max_Combinatorial cluster information and scores
    adata.varm[secondary_features_label + "_Max_Combinatorial_ESSs"] = (
        combinatorial_label_info
    )
    print(
        "Max combinatorial ESSs for given sample labels has been saved as 'adata.varm['"
        + secondary_features_label
        + "_Max_Combinatorial_ESSs"
        + "']'"
    )
    ### Save the new features/clusters that maximise the ESS scores of each feature in adata
    adata.obsm[secondary_features_label + "_Max_ESS_Features"] = csc_matrix(
        top_score_secondary_features.astype("f")
    )
    print(
        "The features/clusters relating to each max_ESS have been saved in 'adata.obsm['"
        + secondary_features_label
        + "_Max_ESS_Features"
        + "']'"
    )
    return adata


def parallel_identify_max_ESSs(secondary_features, sorted_SGs_idxs, use_cores=-1):
    """
    Parallelised version of identify_max_ESSs function
    """
    #
    feature_inds = np.arange(sorted_SGs_idxs.shape[0])
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
    ## Perform calculations
    print("Calculating ESS and EP matricies.")
    print(
        "If progress bar freezes consider increasing system memory or reducing number of cores used with the 'use_cores' parameter as you may have hit a memory ceiling for your machine."
    )
    # if __name__ == '__main__':
    with np.errstate(divide="ignore", invalid="ignore"):
        results = p_map(
            partial(
                identify_max_ESSs,
                secondary_features=secondary_features,
                sorted_SGs_idxs=sorted_SGs_idxs,
            ),
            feature_inds,
            num_cpus=use_cores,
        )
    ## Extract results for each feature in adata
    max_ESSs = np.zeros(len(results)).astype("f")
    max_EPs = np.zeros(len(results)).astype("f")
    top_score_columns_combinations = [[]] * len(results)
    top_score_secondary_features = np.zeros(
        (secondary_features.shape[0], feature_inds.shape[0])
    )
    for i in np.arange(len(results)):
        ESSs = results[i][0]
        EPs = results[i][1]
        max_ESS_idx = np.argmax(ESSs)
        max_ESSs[i] = ESSs[max_ESS_idx]
        max_EPs[i] = EPs[max_ESS_idx]
        top_score_columns_combinations[i] = sorted_SGs_idxs[i, :][
            np.arange(0, max_ESS_idx + 1)
        ].tolist()
        top_score_secondary_features[:, i] = (
            secondary_features[:, top_score_columns_combinations[i]]
            .sum(axis=1)
            .A.reshape(-1)
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
    sort_order = np.delete(sorted_SGs_idxs[FF_ind, :], -1)
    ## From the ordered one-hut clusters, take the cumulative row sums, thereby creating the set of linearly combined one-hot
    # clusters for which we will calculate the ESSs of the fixed feature against.
    secondary_features = np.cumsum(secondary_features.A[:, sort_order], axis=1)
    secondary_features = csc_matrix(secondary_features.astype("f"))
    #### Calculate constants required for ES calculations
    SF_sums = secondary_features.A.sum(axis=0)
    SF_minority_states = SF_sums.copy()
    SF_minority_states[SF_minority_states >= (sample_cardinality / 2)] = (
        sample_cardinality
        - SF_minority_states[SF_minority_states >= (sample_cardinality / 2)]
    )
    ##
    fixed_feature = fixed_feature.reshape(sample_cardinality, 1)  # Might be superfluous
    ## Calculate feature sums
    fixed_feature_cardinality = np.sum(fixed_feature)
    fixed_feature_minority_state = fixed_feature_cardinality.copy()
    if fixed_feature_minority_state >= (sample_cardinality / 2):
        fixed_feature_minority_state = sample_cardinality - fixed_feature_minority_state
    #
    ## Identify where FF is the QF or RF
    FF_QF_vs_RF = np.zeros(SF_minority_states.shape[0])
    FF_QF_vs_RF[np.where(fixed_feature_minority_state > SF_minority_states)[0]] = (
        1  # 1's mean FF is QF
    )
    ## Caclulate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = SF_minority_states.copy()
    idxs = np.where(FF_QF_vs_RF == 0)[0]
    RFms[idxs] = fixed_feature_minority_state
    RFMs = sample_cardinality - RFms
    QFms = SF_minority_states.copy()
    idxs = np.where(FF_QF_vs_RF == 1)[0]
    QFms[idxs] = fixed_feature_minority_state
    QFMs = sample_cardinality - QFms
    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    max_ent_x_mm = (RFms * QFms) / (RFms + RFMs)
    max_ent_x_Mm = (QFMs * RFms) / (RFms + RFMs)
    max_ent_x_mM = (RFMs * QFms) / (RFms + RFMs)
    max_ent_x_MM = (RFMs * QFMs) / (RFms + RFMs)
    max_ent_options = np.array([max_ent_x_mm, max_ent_x_Mm, max_ent_x_mM, max_ent_x_MM])
    ####
    # Caclulate the overlap between the FF states and the secondary features, using the correct ESE (1-4)
    all_use_cases, all_overlaps_options, all_used_inds = (
        identify_max_ESSs_get_overlap_info(
            fixed_feature,
            fixed_feature_cardinality,
            sample_cardinality,
            SF_sums,
            FF_QF_vs_RF,
            secondary_features,
        )
    )
    ## Having extracted the overlaps and their respective ESEs (1-4), calcualte the ESS and EPs
    ESSs, D_EPs, O_EPs, SWs, SGs = calc_ESSs(
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
    # EPs = np.maximum(D_EPs,O_EPs)
    identical_features = np.where(ESSs == 1)[0]
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
    all_use_cases = np.zeros((4, feature_sums.shape[0]))
    ## Set up an array to track the observed overlaps between the FF and the secondary features.
    all_overlaps_options = np.zeros((4, feature_sums.shape[0]))
    ## Set up a list to track the used inds/features for each ESE
    all_used_inds = [[]] * 4
    #
    nonzero_inds = np.where(fixed_feature != 0)[0]
    sub_secondary_features = secondary_features[nonzero_inds, :]
    B = csc_matrix(
        (
            fixed_feature[nonzero_inds].T[0][sub_secondary_features.indices],
            sub_secondary_features.indices,
            sub_secondary_features.indptr,
        )
    )
    overlaps = sub_secondary_features.minimum(B).sum(axis=0).A[0]
    #
    inverse_fixed_feature = np.max(fixed_feature) - fixed_feature
    nonzero_inds = np.where(inverse_fixed_feature != 0)[0]
    sub_secondary_features = secondary_features[nonzero_inds, :]
    B = csc_matrix(
        (
            inverse_fixed_feature[nonzero_inds].T[0][sub_secondary_features.indices],
            sub_secondary_features.indices,
            sub_secondary_features.indptr,
        )
    )
    inverse_overlaps = sub_secondary_features.minimum(B).sum(axis=0).A[0]
    ## If FF is observed in it's minority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality < (sample_cardinality / 2):
        #######
        ## FF and other feature are minority states & FF is QF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as mm (row 1), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = np.array([1, -1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        #######
        ## FF and other feature are minority states & FF is RF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as mm (row 1), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = np.array([1, 0, -1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[0, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is QF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as mM (row 4), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = np.array([0, 0, 1, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        #######
        ## FF is minority, other feature is majority & FF is RF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as Mm (row 2), and which are mm when the secondary feature is flipped (row 1)
        all_use_cases[:, calc_idxs] = np.array([0, 1, 0, -1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[3, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        #
    ## If FF is observed in it's majority state, use the following 4 steps to caclulate overlaps with every other feature
    if fixed_feature_cardinality >= (sample_cardinality / 2):
        #######
        ## FF is majority, other feature is minority & FF is QF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as Mm (row 2), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = np.array([-1, 1, 0, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is minority & FF is RF
        calc_idxs = np.where(
            (feature_sums < (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as mM (row 3), and which are MM when the secondary feature is flipped (row 4)
        all_use_cases[:, calc_idxs] = np.array([-1, 0, 1, 0]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[0, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mm
        all_used_inds[0] = np.append(all_used_inds[0], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is QF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 1)
        )[0]
        ## Track which features are observed as MM (row 4), and which are Mm when the secondary feature is flipped (row 2)
        all_use_cases[:, calc_idxs] = np.array([0, 0, -1, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[2, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_mM
        all_used_inds[2] = np.append(all_used_inds[2], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        #######
        ## FF is majority, other feature is majority & FF is RF
        calc_idxs = np.where(
            (feature_sums >= (sample_cardinality / 2)) & (FF_QF_vs_RF == 0)
        )[0]
        ## Track which features are observed as MM (row 4), and which are mM when the secondary feature is flipped (row 3)
        all_use_cases[:, calc_idxs] = np.array([0, -1, 0, 1]).reshape(4, 1)
        ## Calcualte the overlaps as the sum of minimums between samples, using global_scaled_matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        all_overlaps_options[1, calc_idxs] = inverse_overlaps[calc_idxs]  # Overlaps_Mm
        all_used_inds[1] = np.append(all_used_inds[1], calc_idxs)
        all_overlaps_options[3, calc_idxs] = overlaps[calc_idxs]  # Overlaps_MM
        all_used_inds[3] = np.append(all_used_inds[3], calc_idxs)
        #
    return all_use_cases, all_overlaps_options, all_used_inds


##### Find minimal set of marker genes functions #####


def find_minimal_combinatorial_gene_set(
    adata,
    N,
    secondary_features_label,
    input_genes=np.array([]),
    num_reheats=5,
    resolution=1,
    use_cores=-1,
):
    """
    Having used find_max_ESSs identify a set of features/clusters that maximise the ESS of each variable/column in adata and parallel_calc_es_matrices
    to calculate the ESSs of every varible/column in adata in relation to each ESS_Max feature/cluster, we can now use find_minimal_combinatorial_gene_set
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
    if input_genes.shape[0] == 0:
        input_genes = np.array(adata.var_names.tolist())
    #
    input_gene_idxs = np.where(np.isin(adata.var_names, input_genes))[0]
    ###
    global clust_ESSs
    clust_ESSs = np.asarray(adata.varm[secondary_features_label + "_ESSs"])[
        np.ix_(input_gene_idxs, input_gene_idxs)
    ]
    clust_ESSs[clust_ESSs < 0] = 0
    ###
    max_ESSs = np.max(clust_ESSs, axis=1)
    ###
    chosen_clusts = np.random.choice(clust_ESSs.shape[0], N, replace=False)
    ###
    best_score = -np.inf
    reheat = 0
    while reheat <= num_reheats:
        ###
        chosen_pairwise_ESSs = clust_ESSs[np.ix_(chosen_clusts, chosen_clusts)]
        np.fill_diagonal(chosen_pairwise_ESSs, 0)
        current_score = np.sum(
            max_ESSs[chosen_clusts]
            - (np.max(chosen_pairwise_ESSs, axis=1) * resolution)
        )
        ###
        end = 0
        ###
        while end == 0:
            ###
            all_max_changes, all_max_change_idxs, all_max_replacement_scores = (
                all_max_changes,
                all_max_change_idxs,
                all_max_replacement_scores,
            ) = parallel_replace_clust(
                N,
                chosen_pairwise_ESSs,
                chosen_clusts,
                current_score,
                max_ESSs,
                resolution,
                use_cores=use_cores,
            )
            ###
            replace_clust_idx = np.argmax(all_max_changes)
            if all_max_changes[replace_clust_idx] > 0:
                # print(all_max_replacement_scores[replace_clust_idx])
                replacement_clust_idx = all_max_change_idxs[replace_clust_idx]
                #
                # print(Replacement_Gene_Ind)
                chosen_clusts[replace_clust_idx] = replacement_clust_idx
                #
                chosen_pairwise_ESSs = clust_ESSs[np.ix_(chosen_clusts, chosen_clusts)]
                np.fill_diagonal(chosen_pairwise_ESSs, 0)
                current_score = np.sum(
                    max_ESSs[chosen_clusts]
                    - (np.max(chosen_pairwise_ESSs, axis=1) * resolution)
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
        reheat_num = int(np.ceil(N * 0.25))
        random_reheat_1 = np.random.choice(
            input_genes.shape[0], reheat_num, replace=False
        )
        random_reheat_2 = np.random.randint(chosen_clusts.shape[0], size=reheat_num)
        chosen_clusts[random_reheat_2] = random_reheat_1
        reheat = reheat + 1
        print("reheat number: " + str(reheat))
    #
    chosen_pairwise_ESSs = clust_ESSs[np.ix_(chosen_clusts, chosen_clusts)]
    return best_chosen_clusters, input_genes[best_chosen_clusters], chosen_pairwise_ESSs


def replace_clust(
    replace_idx,
    chosen_pairwise_ESSs,
    chosen_clusts,
    current_score,
    max_ESSs,
    resolution,
):
    sub_chosen_clusts = np.delete(chosen_clusts, replace_idx)
    #
    sub_chosen_pairwise_ESSs = np.delete(
        chosen_pairwise_ESSs, replace_idx, axis=0
    )  # Delete a row, which should be the cluster
    sub_chosen_pairwise_ESSs = np.delete(sub_chosen_pairwise_ESSs, replace_idx, axis=1)
    sub_2nd_maxs = np.max(sub_chosen_pairwise_ESSs, axis=1)
    replacement_columns = clust_ESSs[sub_chosen_clusts, :]
    replacement_columns[(np.arange(sub_chosen_clusts.shape[0]), sub_chosen_clusts)] = 0
    sub_2nd_maxs = np.maximum(sub_2nd_maxs[:, np.newaxis], replacement_columns)
    #
    replacement_scores_1 = np.sum(
        (max_ESSs[sub_chosen_clusts, np.newaxis] - (sub_2nd_maxs * resolution)), axis=0
    )
    #
    replacement_rows = clust_ESSs[:, sub_chosen_clusts]
    replacement_rows[(sub_chosen_clusts, np.arange(sub_chosen_clusts.shape[0]))] = 0
    #
    replacement_scores_2 = max_ESSs - (np.max(replacement_rows, axis=1) * resolution)
    #
    replacement_scores = replacement_scores_1 + replacement_scores_2
    ###
    changes = replacement_scores - current_score
    changes[chosen_clusts] = -np.inf
    max_idx = np.argmax(changes)
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
    replace_idxs = np.arange(N)
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
    results = np.asarray(results)
    all_max_changes = results[:, 0]
    all_max_change_idxs = results[:, 1]
    all_max_replacement_scores = results[:, 2]
    return all_max_changes, all_max_change_idxs, all_max_replacement_scores


#### ESFS workflow plotting functions ####


def knn_Smooth_Gene_Expression(
    adata, use_genes, knn=30, metric="correlation", log_scale=False
):
    #
    print(
        "Calculating pairwise cell-cell distance matrix. Distance metric = "
        + metric
        + ", knn = "
        + str(knn)
    )
    if issparse(adata.X) == True:
        distmat = squareform(pdist(adata[:, use_genes].X.A, metric))
        smoothed_expression = adata.X.A.copy()
    else:
        distmat = squareform(pdist(adata[:, use_genes].X, metric))
        smoothed_expression = adata.X.copy()
    neighbors = np.sort(np.argsort(distmat, axis=1)[:, 0:knn])
    #
    if log_scale == True:
        smoothed_expression = np.log2(smoothed_expression + 1)
    #
    neighbour_expression = smoothed_expression[neighbors]
    smoothed_expression = np.mean(neighbour_expression, axis=1)

    print(
        "A Smoothed_Expression sparse csc_matrix matrix with knn = "
        + str(knn)
        + " has been saved to 'adata.layers['Smoothed_Expression']'"
    )
    adata.layers["Smoothed_Expression"] = csc_matrix(smoothed_expression.astype("f"))
    return adata


def ES_Rank_Genes(
    adata,
    ESS_threshold,
    EP_threshold=0,
    exclude_genes=np.array([]),
    known_important_genes=np.array([]),
    secondary_features_label="Self",
    min_edges=5,
):
    ##
    # ESSs = adata.varp['ESSs']
    masked_ESSs = adata.varm[secondary_features_label + "_ESSs"].copy()
    masked_EPs = adata.varm[secondary_features_label + "_EPs"].copy()
    mask = np.where((masked_EPs < EP_threshold) | (masked_ESSs < ESS_threshold))
    masked_ESSs[mask] = 0
    masked_EPs[mask] = 0
    used_features = np.asarray(adata.var.index)
    used_features_idxs = np.arange(used_features.shape[0])
    ##
    low_connectivity = used_features[
        np.where(np.sum((masked_EPs > 0), axis=0) < min_edges)[0]
    ]
    remove_genes = np.unique(np.append(exclude_genes, low_connectivity))
    ##
    print(
        "Pruning ESS graph by removing genes with with low numbers of edges (min_edges = "
        + str(min_edges)
        + ")"
    )
    print("Starting genes = " + str(used_features_idxs.shape[0]))
    while remove_genes.shape[0] > 0:
        excude_genes_idxs = np.where(np.isin(used_features, remove_genes))[0]
        #
        used_features_idxs = np.delete(used_features_idxs, excude_genes_idxs)
        used_features = np.delete(used_features, excude_genes_idxs)
        # absolute_ESSs = absolute_ESSs[np.ix_(used_features_idxs,used_features_idxs)]
        # ESSs = ESSs[np.ix_(used_features_idxs,used_features_idxs)]
        masked_EPs = masked_EPs[np.ix_(used_features_idxs, used_features_idxs)]
        masked_ESSs = masked_ESSs[np.ix_(used_features_idxs, used_features_idxs)]
        #
        used_features_idxs = np.arange(used_features.shape[0])
        print("Remaining genes = " + str(used_features_idxs.shape[0]))
        remove_genes = used_features[
            np.where(np.sum((masked_EPs > 0), axis=0) < min_edges)[0]
        ]
    ##
    # masked_ESSs = absolute_ESSs.copy()
    # masked_ESSs[np.where((masked_EPs < EP_threshold) | (absolute_ESSs < ESS_threshold))] = 0
    ##
    print("Caclulating feature weights")
    feature_weights = np.average(masked_ESSs, weights=masked_EPs, axis=0)
    sig_genes_per_gene = (masked_EPs > EP_threshold).sum(1)
    norm_network_feature_weights = feature_weights / sig_genes_per_gene
    ##
    if known_important_genes.shape[0] > 0:
        ranks = pd.DataFrame(
            np.zeros((1, known_important_genes.shape[0])),
            index=["Rank"],
            columns=known_important_genes,
        )
        rank_sorted = used_features[np.argsort(-norm_network_feature_weights)]
        for i in np.arange(known_important_genes.shape[0]):
            rank = np.where(rank_sorted == known_important_genes[i])[0]
            if rank.shape[0] > 0:
                ranks[known_important_genes[i]] = rank
            else:
                ranks[known_important_genes[i]] = np.nan
        ##
        print("Known inportant gene ranks:")
        print(ranks)
    ##
    idxs = np.argsort(-norm_network_feature_weights)
    norm_weights = pd.DataFrame(
        norm_network_feature_weights[idxs], index=used_features[idxs]
    )
    ranked_genes = pd.DataFrame(
        np.arange(idxs.shape[0]).astype("i"), index=used_features[idxs]
    )
    ##
    adata.var["ESFS_Gene_Weights"] = norm_weights
    print("ESFS gene weights have been saved to 'adata.var['ESFS_Gene_Weights']'")
    adata.var["ES_Rank"] = ranked_genes
    print("ESFS gene ranks have been saved to 'adata.var['ES_Rank']'")
    ##
    return adata


def plot_top_ranked_genes_UMAP(
    adata,
    top_ranked_genes,
    clustering="None",
    known_important_genes=np.array([]),
    UMAP_min_dist=0.1,
    UMAP_neighbours=20,
    hdbscan_min_cluster_size=50,
    secondary_features_label="Self",
):
    print(
        "Visualising the ESS graph of the top "
        + str(top_ranked_genes)
        + " ranked genes in a UMAP."
    )
    top_ESS_gene_idxs = np.where(adata.var["ES_Rank"] < top_ranked_genes)[0]
    top_ESS_genes = adata.var["ES_Rank"].index[top_ESS_gene_idxs]
    ##
    masked_ESSs = adata.varm[secondary_features_label + "_ESSs"].copy()[
        np.ix_(top_ESS_gene_idxs, top_ESS_gene_idxs)
    ]
    # masked_ESSs[adata.varp["EPs"][np.ix_(Top_ESS_Gene_idxs,Top_ESS_Gene_idxs)] < 0] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gene_embedding = umap.UMAP(
            n_neighbors=UMAP_neighbours,
            min_dist=UMAP_min_dist,
            n_components=2,
            random_state=42,
        ).fit_transform(masked_ESSs)
        # gene_embedding = umap.UMAP(n_neighbors=UMAP_Neighbours, min_dist=UMAP_min_dist, n_components=2).fit_transform(masked_ESSs)
    ##
    # No clustering
    if clustering == "None":
        print(
            "Clustering == 'None', set Clustering to numeric value for Kmeans clustering or 'hdbscan' for automated density clustering."
        )
        plt.figure(figsize=(5, 5))
        plt.title("Top " + str(top_ranked_genes) + " genes UMAP", fontsize=20)
        plt.scatter(gene_embedding[:, 0], gene_embedding[:, 1], s=7)
        plt.xlabel("UMAP 1", fontsize=16)
        plt.ylabel("UMAP 2", fontsize=16)
        labels = np.zeros(top_ranked_genes)
    # Kmeans clustering
    if isinstance(clustering, int):
        print(
            "Clustering == an integer value leading to Kmeans clustering, set Clustering to 'hdbscan' for automated density clustering."
        )
        kmeans = KMeans(n_clusters=clustering, random_state=42, n_init="auto").fit(
            gene_embedding
        )
        labels = kmeans.labels_
        unique_labels = np.unique(labels)
        #
        plt.figure(figsize=(5, 5))
        plt.title(
            "Top " + str(top_ranked_genes) + " genes UMAP\nClustering = Kmeans",
            fontsize=20,
        )
        for i in np.arange(unique_labels.shape[0]):
            idxs = np.where(labels == unique_labels[i])[0]
            plt.scatter(
                gene_embedding[idxs, 0],
                gene_embedding[idxs, 1],
                s=7,
                label=unique_labels[i],
            )
        #
        plt.xlabel("UMAP 1", fontsize=16)
        plt.ylabel("UMAP 2", fontsize=16)
    # hdbscan clustering
    if clustering == "hdbscan":
        print(
            "Clustering == 'hdbscan', set Clustering to an integer value for automated Kmeans clustering."
        )
        hdb = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        np.random.seed(42)
        hdb.fit(gene_embedding)
        labels = hdb.labels_
        unique_labels = np.unique(labels)
        #
        plt.figure(figsize=(5, 5))
        plt.title(
            "Top " + str(top_ranked_genes) + " genes UMAP\nClustering = hdbscan",
            fontsize=20,
        )
        for i in np.arange(unique_labels.shape[0]):
            idxs = np.where(labels == unique_labels[i])[0]
            plt.scatter(
                gene_embedding[idxs, 0],
                gene_embedding[idxs, 1],
                s=7,
                label=unique_labels[i],
            )
        #
        plt.xlabel("UMAP 1", fontsize=16)
        plt.ylabel("UMAP 2", fontsize=16)
    #
    if known_important_genes.shape[0] > 0:
        important_gene_idxs = np.where(np.isin(top_ESS_genes, known_important_genes))[0]
        plt.scatter(
            gene_embedding[important_gene_idxs, 0],
            gene_embedding[important_gene_idxs, 1],
            s=15,
            c="black",
            marker="x",
            label="Known important genes",
        )
    #
    plt.legend()
    print(
        "This function outputs the 'Top_ESS_Genes', 'Gene_Clust_Labels' and 'Gene_Embedding' objects in case users would like to investiage them further."
    )
    return top_ESS_genes, labels, gene_embedding


def get_gene_cluster_cell_UMAPs(
    adata,
    gene_clust_labels,
    top_ESS_genes,
    n_neighbors,
    min_dist,
    log_transformed,
    specific_cluster=None,
):
    print(
        "Generating the cell UMAP embeddings for each cluster of genes from the previous function."
    )
    if specific_cluster == None:
        unique_gene_clust_labels = np.unique(gene_clust_labels)
        gene_cluster_selected_genes = [[]] * unique_gene_clust_labels.shape[0]
        gene_cluster_embeddings = [[]] * unique_gene_clust_labels.shape[0]
    else:
        unique_gene_clust_labels = np.array([specific_cluster])
        gene_cluster_selected_genes = [[]] * unique_gene_clust_labels.shape[0]
        gene_cluster_embeddings = [[]] * unique_gene_clust_labels.shape[0]
    for i in np.arange(unique_gene_clust_labels.shape[0]):
        print(
            "Plotting cell UMAP using gene clusters " + str(unique_gene_clust_labels[i])
        )
        selected_genes = np.asarray(
            top_ESS_genes[
                np.where(np.isin(gene_clust_labels, unique_gene_clust_labels[i]))[0]
            ]
        )
        gene_cluster_selected_genes[i] = selected_genes
        # np.save(path + "Saved_ESFS_Genes.npy",np.asarray(selected_genes))
        reduced_input_data = adata[:, selected_genes].X.A
        if log_transformed == True:
            reduced_input_data = np.log2(reduced_input_data + 1)
        #
        # embedding_model = umap.UMAP(n_neighbors=n_neighbors, metric="correlation",min_dist=min_dist,n_components=2,random_state=42).fit(reduced_input_data)
        embedding_model = umap.UMAP(
            n_neighbors=n_neighbors,
            metric="correlation",
            min_dist=min_dist,
            n_components=2,
        ).fit(reduced_input_data)
        gene_cluster_embeddings[i] = embedding_model.embedding_
        #
    return gene_cluster_embeddings, gene_cluster_selected_genes


def plot_gene_cluster_cell_UMAPs(
    adata,
    gene_cluster_embeddings,
    gene_cluster_selected_genes,
    cell_label="None",
    ncol=1,
    log2_gene_expression=True,
):
    #
    if np.isin(cell_label, adata.obs.columns):
        cell_labels = adata.obs[cell_label]
        unique_cell_labels = np.unique(cell_labels)
        #
        for i in np.arange(len(gene_cluster_embeddings)):
            embedding = gene_cluster_embeddings[i]
            plt.figure(figsize=(7, 5))
            plt.title(
                "Cell UMAP"
                + "\n"
                + str(gene_cluster_selected_genes[i].shape[0])
                + " genes",
                fontsize=20,
            )
            for j in np.arange(unique_cell_labels.shape[0]):
                IDs = np.where(cell_labels == unique_cell_labels[j])
                plt.scatter(
                    embedding[IDs, 0],
                    embedding[IDs, 1],
                    s=3,
                    label=unique_cell_labels[j],
                )
            #
            plt.xlabel("UMAP 1", fontsize=16)
            plt.ylabel("UMAP 2", fontsize=16)
            #
            if cell_label != "None":
                # Adjust legend to be outside and below the plot
                plt.legend(
                    loc="center left",  # Align legend to the left of the bounding box
                    bbox_to_anchor=(
                        1,
                        0.5,
                    ),  # Position to the right and vertically centered
                    ncol=1,  # Keep in a single column to span height
                    fontsize=10,
                    frameon=False,
                    markerscale=5,
                )
    #
    if np.isin(cell_label, adata.var.index):
        #
        expression = adata[:, cell_label].X.T
        if issparse(expression):
            expression = expression.todense()
        #
        expression = np.asarray(expression)[0]
        #
        if log2_gene_expression == True:
            expression = np.log2(expression + 1)
        #
        for i in np.arange(len(gene_cluster_embeddings)):
            embedding = gene_cluster_embeddings[i]
            plt.figure(figsize=(7, 5))
            plt.title("Cell UMAP" + "\n" + cell_label, fontsize=20)
            #
            plt.scatter(
                embedding[:, 0], embedding[:, 1], s=3, c=expression, cmap="seismic"
            )
            #
            plt.xlabel("UMAP 1", fontsize=16)
            plt.ylabel("UMAP 2", fontsize=16)
            cb = plt.colorbar()
            cb.set_label("$log_2$(Expression)", labelpad=-50, fontsize=10)
    #
    if (np.isin(cell_label, adata.obs.columns)) & (
        np.isin(cell_label, adata.var.index)
    ) == False:
        print(
            "Cell label or gene not found in 'adata.obs.columns' or 'adata.var.index'"
        )


# def Convert_To_Ranked_Gene_List(adata,sample_labels):
#     ESSs = adata.varm[sample_labels+'_ESSs']
#     Columns = np.asarray(ESSs.columns)
#     Ranked_Gene_List = pd.DataFrame(np.zeros(ESSs.shape),columns=Columns)
#     for i in np.arange(Columns.shape[0]):
#         Ranked_Gene_List[Columns[i]] = ESSs.index[np.argsort(-ESSs[Columns[i]])]
#     #
#     return Ranked_Gene_List

# def Display_Chosen_Genes_gif(embedding,Chosen_Genes,adata):
#     # Initialize figure and axis
#     fig, ax = plt.subplots(figsize=(6, 6))

#     # Function to update each frame
#     def update(i):
#         ax.clear()  # Clear previous frame
#         Gene = Chosen_Genes[i]
#         Exp = adata[:,Gene].layers["Smoothed_Expression"].A.ravel()
#         Order = np.argsort(Exp)
#         #
#         ax.set_title(Gene, fontsize=22)
#         Vmax = np.percentile(Exp, 99)
#         if Vmax == 0:
#             Vmax = np.max(Exp)
#         scatter = ax.scatter(embedding[Order, 0], embedding[Order, 1], c=Exp[Order], s=2, vmax=Vmax, cmap="seismic")
#         #
#         ax.set_xticks([])
#         ax.set_yticks([])
#         fig.subplots_adjust(0.02, 0.02, 0.98, 0.9)

#         return scatter,
#     #
#     # Create animation
#     Num_Frames = Chosen_Genes.shape[0]
#     ani = animation.FuncAnimation(fig, update, frames=Num_Frames, interval=500)
#     #
#     plt.close("all")
#     #
#     # Save animation as GIF
#     with tempfile.TemporaryDirectory() as tmpdir:
#         gif_path = os.path.join(tmpdir, "gene_expression.gif")
#         ani.save(gif_path, writer=animation.PillowWriter(fps=0.5))
#         # Display the GIF inline in Jupyter
#         with open(gif_path, "rb") as f:
#             display(Image(data=f.read(), format='png'))

######

