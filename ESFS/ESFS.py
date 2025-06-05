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


def Create_Scaled_Matrix(adata,clip_percentile=97.5,log_scale=False):
    """
    Prior to calculates ES metrics, the data will be scaled to have values
    between 0 and 1. 
    """
    # Filter genes with no expression
    Keep_Genes = Keep_Genes = adata.var_names[np.where(adata.X.getnnz(axis=0) > 50)[0]]
    if Keep_Genes.shape[0] < adata.shape[1]:
        print(str(adata.shape[1] - Keep_Genes.shape[0]) + " genes show no expression. Removing them from adata object")
        adata = adata[:,Keep_Genes]
    # Un-sparsify the data for clipping and scaling
    Scaled_Expressions = adata.X.copy()
    if issparse(Scaled_Expressions) == True:
        Scaled_Expressions = np.asarray(Scaled_Expressions.todense())
    # Log scale the data if user requests.
    if log_scale == True:
        Scaled_Expressions = np.log2(Scaled_Expressions+1)
    # Clip exceptionally high gene expression for each gene. Default percentile is the 97.5th.
    Upper = np.percentile(Scaled_Expressions,clip_percentile,axis=0)
    Upper[np.where(Upper == 0)[0]] = np.max(Scaled_Expressions,axis=0)[np.where(Upper == 0)[0]]
    Scaled_Expressions = Scaled_Expressions.clip(max=Upper[None,:])
    # Normalise gene expression between 0 and 1.
    Scaled_Expressions = Scaled_Expressions / Upper
    # Return data as a sparse csc_matrix
    adata.layers["Scaled_Counts"] = csc_matrix(Scaled_Expressions.astype("f"))
    print("Scaled expression matrix has been saved to 'adata.layers['Scaled_Counts']' as a sparse csc_matrix")
    return adata


def Parallel_Calc_ES_Matricies(adata, Secondary_Features_Label="Self", save_matrices=np.array(["ESSs","EPs"]), Use_Cores=-1):
    """
    Using the Entropy Sorting (ES) mathematical framework, we may caclulate the ESS, EP, SW and SG correlation metrics
    outlined in Radley et al. 2023.
    
    This function assumes that the user aims has a set of input features (Secondary_Features_Label) that will be used to pairwise calculate ES metrics
    against each variable (column) of the provided anndata object. When Secondary_Features_Label is left blank, it defaults
    to "Self" meaning ES metrics will be calcualted pairwise between all of the variables in adata. If Secondary_Features_Label
    is not "Self", the user must point the algorithm to an attribute of adata that contains an array with the same number of samples
    as adata and a set of secondary features (columns).

    save_matrices disctates which ES metrics will be written to the outputted adata object. The options are "ESSs", "EPs", "SGs", "SWs".

    Use_Cores defines how many CPU cores to use. Is Use_Cores = -1, the software will use N-1 the number of cores available on the machine.
    """
    ## Establish which Secondary_Features will be compared against each of the features in adata
    global Secondary_Features
    if Secondary_Features_Label == "Self":
        Secondary_Features = adata.layers["Scaled_Counts"].copy()
    else:
        print("You have provided a 'Secondary_Features_Label', implying that in the anndata object there is a corresponding csc_sparse martix object with rows as samples and columns as features. Each feature will be used to calculate ES scores for each of the variables of the adata object")
        Secondary_Features = adata.obsm[Secondary_Features_Label]
    #
    ## Create the global Global_Scaled_Matrix array for faster parallel computing calculations
    global Global_Scaled_Matrix
    Global_Scaled_Matrix = adata.layers["Scaled_Counts"]
    ## Extract sample and feature cardinality
    Sample_Cardinality = Global_Scaled_Matrix.shape[0]
    ## Calculate feature sums and minority states for each adata feature
    Feature_Sums = Global_Scaled_Matrix.sum(axis=0).A[0]
    Minority_States = Feature_Sums.copy()
    Switch = np.where(Minority_States >= (Sample_Cardinality/2))[0]
    Minority_States[Switch] = Sample_Cardinality - Minority_States[Switch]
    ####
    ## Provide indicies for parallel computing.
    Feature_Inds = np.arange(Secondary_Features.shape[1])
    ## Identify number of cores to use.
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    ## Perform calculations
    print("Calculating ESS and EP matricies.")
    print("If progress bar freezes consider increasing system memory or reducing number of cores used with the 'Use_Cores' parameter as you may have hit a memory ceiling for your machine.")
    # if __name__ == '__main__':
    ## Parallel compute 
    with np.errstate(divide='ignore',invalid='ignore'):
        Results = p_map(partial(Calc_ES_Metrics,Sample_Cardinality=Sample_Cardinality,Feature_Sums=Feature_Sums,Minority_States=Minority_States), Feature_Inds, num_cpus=Use_Cores)
    ## Unpack results
    Results = np.asarray(Results)
    ## Save outputs requested by the save_matrices paramater
    if np.isin("ESSs",save_matrices):
        ESSs = Results[:,0,:]
        if Secondary_Features_Label == "Self": ## The vast majority of outputs are symmetric, but float errors appear to make some non-symmetric. If we can fix this that could be cool.
            ESSs = nanmaximum(ESSs, ESSs.T)
        #
        # Label_ESSs = pd.DataFrame(ESSs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[Secondary_Features_Label + "_ESSs"] = ESSs.T
        print("ESSs for " + Secondary_Features_Label + " label have been saved to " + "'adata.varm['" + Secondary_Features_Label + "_ESSs']'")
        #
    if np.isin("EPs",save_matrices):
        EPs = Results[:,1,:]
        if Secondary_Features_Label == "Self":
            EPs = nanmaximum(EPs, EPs.T)
        #
        # Label_EPs = pd.DataFrame(EPs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[Secondary_Features_Label + "_EPs"] = EPs.T
        print("EPs for " + Secondary_Features_Label + " label have been saved to " + "'adata.varm['" + Secondary_Features_Label + "_EPs']'")
        #
    if np.isin("SWs",save_matrices):
        SWs = Results[:,2,:]
        if Secondary_Features_Label == "Self":
            SWs = nanmaximum(SWs, SWs.T)
        #
        # Label_SWs = pd.DataFrame(SWs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[Secondary_Features_Label + "_SWs"] = SWs.T
        print("SWs for " + Secondary_Features_Label + " label have been saved to " + "'adata.varm['" + Secondary_Features_Label + "_SWs']'")
        #
    if np.isin("SGs",save_matrices):
        SGs = Results[:,3,:]
        if Secondary_Features_Label == "Self":
            SGs = nanmaximum(SGs, SGs.T)
        #
        # Label_SGs = pd.DataFrame(SGs.T,columns=Fixed_Features.index,index=adata.var.index.tolist())
        adata.varm[Secondary_Features_Label + "_SGs"] = SGs.T
        print("SGs for " + Secondary_Features_Label + " label have been saved to " + "'adata.varm['" + Secondary_Features_Label + "_SGs']'")
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
    arr1 = np.where(np.isinf(arr1), -np.inf, arr1)
    arr2 = np.where(np.isinf(arr2), -np.inf, arr2)
    # Replace NaN values with -infinity for comparison
    arr1_nan = np.where(np.isnan(arr1), -np.inf, arr1)
    arr2_nan = np.where(np.isnan(arr2), -np.inf, arr2)
    # Compute the element-wise maximum
    result = np.maximum(arr1_nan, arr2_nan)
    # Where both values are NaN, the result should be NaN
    nan_mask = np.isnan(arr1) & np.isnan(arr2)
    result[nan_mask] = np.nan
    return result


def Calc_ES_Metrics(Feature_Ind,Sample_Cardinality,Feature_Sums,Minority_States):
    """
    This function calcualtes the ES metrics for one of the features in the Secondary_Features object against
    every variable/feature in the adata object.
    
    Feature_Ind - Indicates the column of Secondary_Features being used.
    Sample_Cardinality - Inherited scalar of the number of samples in adata.
    Feature_Sums - Inherited vector of the columns sums of adata.
    Minority_States - Inherited vector of the minority state sums of each column of adata.
    """
    ## Extract the Fixed Feature (FF)
    Fixed_Feature = Secondary_Features[:,Feature_Ind].A
    Fixed_Feature_Cardinality = np.sum(Fixed_Feature)
    Fixed_Feature_Minority_State = Fixed_Feature_Cardinality
    if Fixed_Feature_Minority_State >= (Sample_Cardinality/2):
        Fixed_Feature_Minority_State = Sample_Cardinality - Fixed_Feature_Minority_State
    ## Identify where FF is the QF or RF
    FF_QF_Vs_RF = np.zeros(Feature_Sums.shape[0])
    FF_QF_Vs_RF[np.where(Fixed_Feature_Minority_State > Minority_States)[0]] = 1 # 1's mean FF is QF
    ## Caclulate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = Minority_States.copy()
    Switch = np.where(FF_QF_Vs_RF == 0)[0]
    RFms[Switch] = Fixed_Feature_Minority_State
    RFMs = Sample_Cardinality - RFms
    QFms = Minority_States.copy()
    Switch = np.where(FF_QF_Vs_RF == 1)[0]
    QFms[Switch] = Fixed_Feature_Minority_State
    QFMs = Sample_Cardinality - QFms
    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    Max_Ent_x_mm = (RFms * QFms)/(RFms + RFMs)
    Max_Ent_x_Mm = (QFMs * RFms)/(RFms + RFMs)
    Max_Ent_x_mM = (RFMs * QFms)/(RFms + RFMs)
    Max_Ent_x_MM = (RFMs * QFMs)/(RFms + RFMs)
    Max_Ent_Options = np.array([Max_Ent_x_mm,Max_Ent_x_Mm,Max_Ent_x_mM,Max_Ent_x_MM])
    ######
    ## Caclulate the overlap between the FF states and the secondary features, using the correct ESE (1-4)
    All_Use_Cases, All_Overlaps_Options, All_Used_Inds = Get_Overlap_Info(Fixed_Feature,Fixed_Feature_Cardinality,Sample_Cardinality,Feature_Sums,FF_QF_Vs_RF)
    ## Having extracted the overlaps and their respective ESEs (1-4), calcualte the ESS and EPs
    ESSs, D_EPs, O_EPs, SWs, SGs = Calc_ESSs(RFms, QFms, RFMs, QFMs, Max_Ent_Options, Sample_Cardinality, All_Overlaps_Options, All_Use_Cases, All_Used_Inds)
    Identical_Features = np.where(ESSs==1)[0]
    D_EPs[Identical_Features] = 0
    O_EPs[Identical_Features] = 0
    EPs = nanmaximum(D_EPs, O_EPs)
    return ESSs, EPs, SWs, SGs


def Get_Overlap_Info(Fixed_Feature,Fixed_Feature_Cardinality,Sample_Cardinality,Feature_Sums,FF_QF_Vs_RF):
    """
    For any pair of features the ES mathematical framework has a set of logical rules regarding how the ES metrics
    should be calcualted. These logical rules dictate which of the two features is the reference feature (RF) or query feature
    (QF) and which of the 4 Entropy Sort Equations (ESE 1-4) should be used (Add reference to supplemental figure of 
    new manuscript when ready).
    """
    ## Set up an array to track which of ESE equations 1-4 the recorded observed overlap relates to (row), and if it is
    # native correlation (1) or flipped anti-correlation (-1). Row 1 = mm, row 2 = Mm, row 3 = mM, row 4 = MM.
    All_Use_Cases = np.zeros((4,Feature_Sums.shape[0]))
    ## Set up an array to track the observed overlaps between the FF and the secondary features.
    All_Overlaps_Options = np.zeros((4,Feature_Sums.shape[0]))
    ## Set up a list to track the used inds/features for each ESE
    All_Used_Inds = [[]] * 4
    ####
    ## Pairwise calculate total overlaps of FF values with the values every other a feature in adata
    Non_Zero_Inds = np.where(Fixed_Feature != 0)[0]
    Sub_Global_Scaled_Matrix = Global_Scaled_Matrix[Non_Zero_Inds,:]
    B = csc_matrix((Fixed_Feature[Non_Zero_Inds].T[0][Sub_Global_Scaled_Matrix.indices], Sub_Global_Scaled_Matrix.indices, Sub_Global_Scaled_Matrix.indptr))
    Overlaps = Sub_Global_Scaled_Matrix.minimum(B).sum(axis=0).A[0]
    ## Pairwise calculate total overlaps of Inverse FF values with the values every other a feature in adata
    Inverse_Fixed_Feature = 1 - Fixed_Feature #np.max(Fixed_Feature) - Fixed_Feature
    Non_Zero_Inds = np.where(Inverse_Fixed_Feature != 0)[0]
    Sub_Global_Scaled_Matrix = Global_Scaled_Matrix[Non_Zero_Inds,:]
    B = csc_matrix((Inverse_Fixed_Feature[Non_Zero_Inds].T[0][Sub_Global_Scaled_Matrix.indices], Sub_Global_Scaled_Matrix.indices, Sub_Global_Scaled_Matrix.indptr))
    Inverse_Overlaps = Sub_Global_Scaled_Matrix.minimum(B).sum(axis=0).A[0]
    ####
    ### Using the logical rules of ES to work out which ESE should be used for each pair of features beign compared.
    ## If FF is observed in it's minority state, use the following 4 steps to caclulate overlaps with every other feature
    if (Fixed_Feature_Cardinality < (Sample_Cardinality / 2)):
        #######
        ## FF and other feature are minority states & FF is QF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as mm (row 1), and which are mM when the secondary feature is flipped (row 3)
        All_Use_Cases[:,Calc_Inds] = np.array([1,-1,0,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        All_Overlaps_Options[1,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        #######
        ## FF and other feature are minority states & FF is RF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as mm (row 1), and which are Mm when the secondary feature is flipped (row 2)
        All_Use_Cases[:,Calc_Inds] = np.array([1,0,-1,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        All_Overlaps_Options[2,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        #######
        ## FF is minority, other feature is majority & FF is QF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as mM (row 4), and which are mm when the secondary feature is flipped (row 1)
        All_Use_Cases[:,Calc_Inds] = np.array([0,0,1,-1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[3,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        All_Overlaps_Options[2,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        #######
        ## FF is minority, other feature is majority & FF is RF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as Mm (row 2), and which are mm when the secondary feature is flipped (row 1)
        All_Use_Cases[:,Calc_Inds] = np.array([0,1,0,-1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[3,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        All_Overlaps_Options[1,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        #
    ## If FF is observed in it's majority state, use the following 4 steps to caclulate overlaps with every other feature
    if (Fixed_Feature_Cardinality >= (Sample_Cardinality / 2)):
        #######
        ## FF is majority, other feature is minority & FF is QF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as Mm (row 2), and which are MM when the secondary feature is flipped (row 4)
        All_Use_Cases[:,Calc_Inds] = np.array([-1,1,0,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[1,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        All_Overlaps_Options[0,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        #######
        ## FF is majority, other feature is minority & FF is RF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as mM (row 3), and which are MM when the secondary feature is flipped (row 4)
        All_Use_Cases[:,Calc_Inds] = np.array([-1,0,1,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[2,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        All_Overlaps_Options[0,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        #######
        ## FF is majority, other feature is majority & FF is QF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as MM (row 4), and which are Mm when the secondary feature is flipped (row 2)
        All_Use_Cases[:,Calc_Inds] = np.array([0,0,-1,1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[2,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        All_Overlaps_Options[3,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        #######
        ## FF is majority, other feature is majority & FF is RF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as MM (row 4), and which are mM when the secondary feature is flipped (row 3)
        All_Use_Cases[:,Calc_Inds] = np.array([0,-1,0,1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[1,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        All_Overlaps_Options[3,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        #
    return All_Use_Cases, All_Overlaps_Options, All_Used_Inds


def Calc_ESSs(RFms, QFms, RFMs, QFMs, Max_Ent_Options, Sample_Cardinality, All_Overlaps_Options, All_Use_Cases, All_Used_Inds):
    """
    Now that we have all of the values required for ES caclulations (RFms, QFms, RFMs, QFMs, Max_Ent_Options) and
    have determined which ESE should be used for each pair of features (All_Overlaps_Options, All_Use_Cases, All_Used_Inds),
    we may calculated the ES metrics for the FF against every other feature in adata.
    """
    ## Create variables to track caclulation outputs
    All_ESSs = np.zeros((4,RFms.shape[0]))
    All_D_EPs = np.zeros((4,RFms.shape[0]))
    All_O_EPs = np.zeros((4,RFms.shape[0]))
    All_SGs = np.zeros((4,RFms.shape[0]))
    All_SWs = np.zeros((4,RFms.shape[0]))
    ###################
    ##### (1)  mm #####
    Use_Curve = 0
    ## Find the FF/SF pairs where we should use ESE (1) to calculate entropies
    Calc_Inds = All_Used_Inds[Use_Curve].astype("i")
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Min_Overlap = np.repeat(0,Calc_Inds.shape[0])
        Max_Overlap = RFms[Calc_Inds]
        Overlaps = All_Overlaps_Options[Use_Curve,Calc_Inds]
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Ind_X_1 = Max_Ent_x - Min_Overlap
        Ind_X1 = Max_Overlap - Max_Ent_x
        #
        SD_1_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD1_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SDs = np.zeros(Calc_Inds.shape[0]) - 1
        SDs[SD1_Inds] = 1
        #
        D = np.zeros(Calc_Inds.shape[0])
        O = np.zeros(Calc_Inds.shape[0])
        D[SD_1_Inds] = Overlaps[SD_1_Inds]
        O[SD_1_Inds] = Sample_Cardinality - (RFms[Calc_Inds][SD_1_Inds] + QFms[Calc_Inds][SD_1_Inds]) + Overlaps[SD_1_Inds]
        D[SD1_Inds] = RFms[Calc_Inds][SD1_Inds] - Overlaps[SD1_Inds]
        O[SD1_Inds] = QFms[Calc_Inds][SD1_Inds] - Overlaps[SD1_Inds]
        # Perform caclulations with ESE (1)
        CE, Ind_E, Min_E = ESE1(Overlaps,SDs,RFms[Calc_Inds],RFMs[Calc_Inds],QFms[Calc_Inds],QFMs[Calc_Inds],Sample_Cardinality,Max_Overlap)
        #
        SWs = ((Ind_E-Min_E)/Ind_E)
        SGs = ((Ind_E-CE)/(Ind_E-Min_E))
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < Min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > Ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * All_Use_Cases[Use_Curve,Calc_Inds]
        All_ESSs[Use_Curve,Calc_Inds] = ESS
        All_SWs[Use_Curve,Calc_Inds] = SWs
        All_SGs[Use_Curve,Calc_Inds] = SGs
        #
        SD_1_IndEnt = Ind_E[SD_1_Inds]/Ind_X_1[SD_1_Inds]
        SD1_IndEnt = Ind_E[SD1_Inds]/Ind_X1[SD1_Inds]
        #
        D_EPs = np.zeros(Ind_E.shape[0])
        D_EPs[SD_1_Inds] = ((CE[SD_1_Inds]-Min_E[SD_1_Inds])/D[SD_1_Inds]) - SD_1_IndEnt
        D_EPs[SD1_Inds] = ((CE[SD1_Inds]-Min_E[SD1_Inds])/D[SD1_Inds]) - SD1_IndEnt
        #
        O_EPs = np.zeros(Ind_E.shape[0])
        O_EPs[SD_1_Inds] = ((CE[SD_1_Inds])/O[SD_1_Inds]) - SD_1_IndEnt
        O_EPs[SD1_Inds] = ((CE[SD1_Inds])/O[SD1_Inds]) - SD1_IndEnt
        #
        All_D_EPs[Use_Curve,Calc_Inds] = D_EPs
        All_O_EPs[Use_Curve,Calc_Inds] = O_EPs
        #
    ###################
    ##### (2)  Mm #####
    Use_Curve = 1
    ## Find the FF/SF pairs where we should use ESE (2) to calculate entropies
    Calc_Inds = All_Used_Inds[Use_Curve].astype("i")
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Min_Overlap = np.repeat(0,Calc_Inds.shape[0])
        Max_Overlap = np.minimum(RFms[Calc_Inds],QFMs[Calc_Inds])
        Overlaps =  All_Overlaps_Options[Use_Curve,Calc_Inds]
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Ind_X_1 = Max_Ent_x - Min_Overlap
        Ind_X1 = Max_Overlap - Max_Ent_x
        #
        SD_1_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD1_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SDs = np.zeros(Calc_Inds.shape[0]) - 1
        SDs[SD1_Inds] = 1
        #
        D = np.zeros(Calc_Inds.shape[0])
        O = np.zeros(Calc_Inds.shape[0])
        D[SD_1_Inds] = Overlaps[SD_1_Inds]
        O[SD_1_Inds] = QFms[Calc_Inds][SD_1_Inds] - RFms[Calc_Inds][SD_1_Inds] + Overlaps[SD_1_Inds]
        D[SD1_Inds] = RFms[Calc_Inds][SD1_Inds] - Overlaps[SD1_Inds]
        O[SD1_Inds] = QFMs[Calc_Inds][SD1_Inds] - RFms[Calc_Inds][SD1_Inds] + D[SD1_Inds]
        # Perform caclulations with ESE (2)
        CE, Ind_E, Min_E = ESE2(Overlaps,SDs,RFms[Calc_Inds],RFMs[Calc_Inds],QFms[Calc_Inds],QFMs[Calc_Inds],Sample_Cardinality)
        #
        SWs = ((Ind_E-Min_E)/Ind_E)
        SGs = ((Ind_E-CE)/(Ind_E-Min_E))
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < Min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > Ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * All_Use_Cases[Use_Curve,Calc_Inds]
        All_ESSs[Use_Curve,Calc_Inds] = ESS
        All_SWs[Use_Curve,Calc_Inds] = SWs
        All_SGs[Use_Curve,Calc_Inds] = SGs
        #
        SD_1_IndEnt = Ind_E[SD_1_Inds]/Ind_X_1[SD_1_Inds]
        SD1_IndEnt = Ind_E[SD1_Inds]/Ind_X1[SD1_Inds]
        #
        D_EPs = np.zeros(Ind_E.shape[0])
        D_EPs[SD_1_Inds] = ((CE[SD_1_Inds]-Min_E[SD_1_Inds])/D[SD_1_Inds]) - SD_1_IndEnt
        D_EPs[SD1_Inds] = ((CE[SD1_Inds]-Min_E[SD1_Inds])/D[SD1_Inds]) - SD1_IndEnt
        #
        O_EPs = np.zeros(Ind_E.shape[0])
        O_EPs[SD_1_Inds] = ((CE[SD_1_Inds])/O[SD_1_Inds]) - SD_1_IndEnt
        O_EPs[SD1_Inds] = ((CE[SD1_Inds])/O[SD1_Inds]) - SD1_IndEnt
        #
        All_D_EPs[Use_Curve,Calc_Inds] = D_EPs
        All_O_EPs[Use_Curve,Calc_Inds] = O_EPs
        #
    ###################
    ##### (3)  mM #####
    Use_Curve = 2
    ## Find the FF/SF pairs where we should use ESE (3) to calculate entropies
    Calc_Inds = All_Used_Inds[Use_Curve].astype("i")
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Min_Overlap = RFMs[Calc_Inds]-QFMs[Calc_Inds]
        Max_Overlap = np.minimum(QFms[Calc_Inds],RFMs[Calc_Inds])
        Overlaps =  All_Overlaps_Options[Use_Curve,Calc_Inds]
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Ind_X_1 = Max_Ent_x - Min_Overlap
        Ind_X1 = Max_Overlap - Max_Ent_x
        #
        SD_1_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD1_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SDs = np.zeros(Calc_Inds.shape[0]) - 1
        SDs[SD1_Inds] = 1
        #
        D = np.zeros(Calc_Inds.shape[0])
        O = np.zeros(Calc_Inds.shape[0])
        D[SD_1_Inds] = QFMs[Calc_Inds][SD_1_Inds] - RFMs[Calc_Inds][SD_1_Inds] + Overlaps[SD_1_Inds]
        O[SD_1_Inds] = Overlaps[SD_1_Inds]
        D[SD1_Inds] = QFms[Calc_Inds][SD1_Inds] - Overlaps[SD1_Inds]
        O[SD1_Inds] = RFMs[Calc_Inds][SD1_Inds] - Overlaps[SD1_Inds]
        # Perform caclulations with ESE (3)
        CE, Ind_E, Min_E = ESE3(Overlaps,SDs,RFms[Calc_Inds],RFMs[Calc_Inds],QFms[Calc_Inds],QFMs[Calc_Inds],Sample_Cardinality,Min_Overlap)
        #
        SWs = ((Ind_E-Min_E)/Ind_E)
        SGs = ((Ind_E-CE)/(Ind_E-Min_E))
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < Min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > Ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * All_Use_Cases[Use_Curve,Calc_Inds]
        All_ESSs[Use_Curve,Calc_Inds] = ESS
        All_SWs[Use_Curve,Calc_Inds] = SWs
        All_SGs[Use_Curve,Calc_Inds] = SGs
        #
        SD_1_IndEnt = Ind_E[SD_1_Inds]/Ind_X_1[SD_1_Inds]
        SD1_IndEnt = Ind_E[SD1_Inds]/Ind_X1[SD1_Inds]
        #
        D_EPs = np.zeros(Ind_E.shape[0])
        D_EPs[SD_1_Inds] = ((CE[SD_1_Inds]-Min_E[SD_1_Inds])/D[SD_1_Inds]) - SD_1_IndEnt
        D_EPs[SD1_Inds] = ((CE[SD1_Inds]-Min_E[SD1_Inds])/D[SD1_Inds]) - SD1_IndEnt
        #
        O_EPs = np.zeros(Ind_E.shape[0])
        O_EPs[SD_1_Inds] = ((CE[SD_1_Inds])/O[SD_1_Inds]) - SD_1_IndEnt
        O_EPs[SD1_Inds] = ((CE[SD1_Inds])/O[SD1_Inds]) - SD1_IndEnt
        #
        All_D_EPs[Use_Curve,Calc_Inds] = D_EPs
        All_O_EPs[Use_Curve,Calc_Inds] = O_EPs
        #
    ###################
    ##### (4)  MM #####
    Use_Curve = 3
    ## Find the FF/SF pairs where we should use ESE (4) to calculate entropies
    Calc_Inds = All_Used_Inds[Use_Curve].astype("i")
    if Calc_Inds.shape[0] > 0:
        # Retrieve the Max_Ent, Min_x, Max_X and observed overlap values
        Min_Overlap = QFMs[Calc_Inds]-RFms[Calc_Inds]
        Max_Overlap = np.minimum(QFMs[Calc_Inds],RFMs[Calc_Inds])
        Overlaps =  All_Overlaps_Options[Use_Curve,Calc_Inds]
        Max_Ent_x = Max_Ent_Options[Use_Curve,Calc_Inds]
        Ind_X_1 = Max_Ent_x - Min_Overlap
        Ind_X1 = Max_Overlap - Max_Ent_x
        #
        SD_1_Inds = np.where(Overlaps < Max_Ent_x)[0]
        SD1_Inds = np.where(Overlaps >= Max_Ent_x)[0]
        SDs = np.zeros(Calc_Inds.shape[0]) - 1
        SDs[SD1_Inds] = 1
        #
        D = np.zeros(Calc_Inds.shape[0])
        O = np.zeros(Calc_Inds.shape[0])
        D[SD_1_Inds] = Overlaps[SD_1_Inds] - (Sample_Cardinality - (QFms[Calc_Inds][SD_1_Inds] + RFms[Calc_Inds][SD_1_Inds]))
        O[SD_1_Inds] = Overlaps[SD_1_Inds]
        D[SD1_Inds] = QFMs[Calc_Inds][SD1_Inds] - Overlaps[SD1_Inds]
        O[SD1_Inds] = RFMs[Calc_Inds][SD1_Inds] - QFMs[Calc_Inds][SD1_Inds] + D[SD1_Inds]
        # Perform caclulations with ESE (4)
        CE, Ind_E, Min_E = ESE4(Overlaps,SDs,RFms[Calc_Inds],RFMs[Calc_Inds],QFms[Calc_Inds],QFMs[Calc_Inds],Sample_Cardinality,Min_Overlap,Max_Overlap)
        #
        SWs = ((Ind_E-Min_E)/Ind_E)
        SGs = ((Ind_E-CE)/(Ind_E-Min_E))
        # Because of float errors the following inequality at the boundary sometimes fails, (CE[Test] < Min_E[Test]), leading to values greater than 1. It is valid to just correct them to 1.
        SGs[SGs > 1] = 1
        # Because of float errors the following inequality at the maximum sometimes fails, (CE[Test] > Ind_E[Test]), leading to values greater than less than 0. It is valid to just correct them to 0.
        SGs[SGs < 0] = 0
        #
        ESS = SWs * SGs * SDs * All_Use_Cases[Use_Curve,Calc_Inds]
        All_ESSs[Use_Curve,Calc_Inds] = ESS
        All_SWs[Use_Curve,Calc_Inds] = SWs
        All_SGs[Use_Curve,Calc_Inds] = SGs
        #
        SD_1_IndEnt = Ind_E[SD_1_Inds]/Ind_X_1[SD_1_Inds]
        SD1_IndEnt = Ind_E[SD1_Inds]/Ind_X1[SD1_Inds]
        #
        D_EPs = np.zeros(Ind_E.shape[0])
        D_EPs[SD_1_Inds] = ((CE[SD_1_Inds]-Min_E[SD_1_Inds])/D[SD_1_Inds]) - SD_1_IndEnt
        D_EPs[SD1_Inds] = ((CE[SD1_Inds]-Min_E[SD1_Inds])/D[SD1_Inds]) - SD1_IndEnt
        #
        O_EPs = np.zeros(Ind_E.shape[0])
        O_EPs[SD_1_Inds] = ((CE[SD_1_Inds])/O[SD_1_Inds]) - SD_1_IndEnt
        O_EPs[SD1_Inds] = ((CE[SD1_Inds])/O[SD1_Inds]) - SD1_IndEnt
        #
        All_D_EPs[Use_Curve,Calc_Inds] = D_EPs
        All_O_EPs[Use_Curve,Calc_Inds] = O_EPs
        #
    ########
    ## For each feature pair, accept the orientation with the maximum ESS as it is the least likely to have occoured by chance.
    Max_ESS_Inds = np.nanargmax(np.absolute(All_ESSs),axis=0)
    ## Return results
    return All_ESSs[Max_ESS_Inds,np.arange(RFms.shape[0])], All_D_EPs[Max_ESS_Inds,np.arange(RFms.shape[0])], All_O_EPs[Max_ESS_Inds,np.arange(RFms.shape[0])], All_SWs[Max_ESS_Inds,np.arange(RFms.shape[0])], All_SGs[Max_ESS_Inds,np.arange(RFms.shape[0])]


def ESE1(x,SD,RFm,RFM,QFm,QFM,Ts,Max_Overlap):
    """
    This function takes the observed inputs and uses the ESE1 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (Ind_E) and Minimum Entropy (Min_E).
    """
    G1_E = (RFm/Ts) * ((((x)/RFm)*(-np.log((x)/RFm))) + (((RFm-x)/RFm)*(-np.log((RFm-x)/RFm))))
    G2_E = (RFM/Ts)*((((QFm-x)/RFM)*(-np.log((QFm-x)/RFM))) + (((RFM-QFm+x)/RFM)*(-np.log((RFM-QFm+x)/RFM))))
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    Ind_E = (QFm/Ts)*(-np.log((QFm/Ts))) + (QFM/Ts)*(-np.log((QFM/Ts)))
    #
    Min_E = np.zeros(SD.shape[0])
    SD_1_Inds = np.where(SD == -1)[0]
    Min_E[SD_1_Inds] = (RFM[SD_1_Inds]/Ts)*((((QFm[SD_1_Inds])/RFM[SD_1_Inds])*(-np.log((QFm[SD_1_Inds])/RFM[SD_1_Inds]))) + (((RFM[SD_1_Inds]-QFm[SD_1_Inds])/RFM[SD_1_Inds])*(-np.log((RFM[SD_1_Inds]-QFm[SD_1_Inds])/RFM[SD_1_Inds]))))
    SD1_Inds = np.where(SD == 1)[0]
    Min_E[SD1_Inds] = (RFM[SD1_Inds]/Ts)*((((QFm[SD1_Inds]-Max_Overlap[SD1_Inds])/RFM[SD1_Inds])*(-np.log((QFm[SD1_Inds]-Max_Overlap[SD1_Inds])/RFM[SD1_Inds]))) + (((RFM[SD1_Inds]-QFm[SD1_Inds]+Max_Overlap[SD1_Inds])/RFM[SD1_Inds])*(-np.log((RFM[SD1_Inds]-QFm[SD1_Inds]+Max_Overlap[SD1_Inds])/RFM[SD1_Inds]))))
    Min_E[np.isnan(Min_E)] = 0
    Min_E[np.isnan(Min_E)] = 0
    #
    CE[np.isnan(CE)] = Min_E[np.isnan(CE)]
    return CE, Ind_E, Min_E

def ESE2(x,SD,RFm,RFM,QFm,QFM,Ts):
    """
    This function takes the observed inputs and uses the ESE2 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (Ind_E) and Minimum Entropy (Min_E).
    """
    G1_E = (RFm/Ts)*((-(((RFm-x)/RFm)*np.log((RFm-x)/RFm))-(((x)/RFm)*np.log((x)/RFm))))
    G2_E = (RFM/Ts)*((-(((RFM-QFM+x)/RFM)*np.log((RFM-QFM+x)/RFM))-(((QFM-x)/RFM)*np.log((QFM-x)/RFM))))
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    Ind_E = (QFm/Ts)*(-np.log((QFm/Ts))) + (QFM/Ts)*(-np.log((QFM/Ts)))
    #
    Min_E = np.zeros(SD.shape[0])
    SD_1_Inds = np.where(SD == -1)[0]
    Min_E[SD_1_Inds] = (RFM[SD_1_Inds]/Ts)*((-(((RFM[SD_1_Inds]-QFM[SD_1_Inds])/RFM[SD_1_Inds])*np.log((RFM[SD_1_Inds]-QFM[SD_1_Inds])/RFM[SD_1_Inds]))-(((QFM[SD_1_Inds])/RFM[SD_1_Inds])*np.log((QFM[SD_1_Inds])/RFM[SD_1_Inds]))))
    SD1_Inds = np.where(SD == 1)[0]
    Min_E[SD1_Inds] = (RFM[SD1_Inds]/Ts)*((-(((RFM[SD1_Inds]-QFM[SD1_Inds]+RFm[SD1_Inds])/RFM[SD1_Inds])*np.log((RFM[SD1_Inds]-QFM[SD1_Inds]+RFm[SD1_Inds])/RFM[SD1_Inds]))-(((QFM[SD1_Inds]-RFm[SD1_Inds])/RFM[SD1_Inds])*np.log((QFM[SD1_Inds]-RFm[SD1_Inds])/RFM[SD1_Inds]))))
    Min_E[np.isnan(Min_E)] = 0
    #
    CE[np.isnan(CE)] = Min_E[np.isnan(CE)]
    return CE, Ind_E, Min_E

def ESE3(x,SD,RFm,RFM,QFm,QFM,Ts,Min_Overlap):
    """
    This function takes the observed inputs and uses the ESE3 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (Ind_E) and Minimum Entropy (Min_E).
    """
    G1_E = (RFm/Ts)*((-(((QFm-x)/RFm)*np.log((QFm-x)/RFm))-(((RFm-QFm+x)/RFm)*np.log((RFm-QFm+x)/RFm))))
    G2_E = (RFM/Ts)*((-(((x)/RFM)*np.log((x)/RFM))-(((RFM-x)/RFM)*np.log((RFM-x)/RFM))))
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    Ind_E = (QFm/Ts)*(-np.log((QFm/Ts))) + (QFM/Ts)*(-np.log((QFM/Ts)))
    #
    Min_E = np.zeros(SD.shape[0])
    SD_1_Inds = np.where(SD == -1)[0]
    Min_E[SD_1_Inds] = (RFM[SD_1_Inds]/Ts)*((-(((Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds])*np.log((Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds]))-(((RFM[SD_1_Inds]-Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds])*np.log((RFM[SD_1_Inds]-Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds]))))
    SD1_Inds = np.where(SD == 1)[0]
    Min_E[SD1_Inds] = (RFM[SD1_Inds]/Ts)*((-(((RFM[SD1_Inds]-QFM[SD1_Inds]+RFm[SD1_Inds])/RFM[SD1_Inds])*np.log((RFM[SD1_Inds]-QFM[SD1_Inds]+RFm[SD1_Inds])/RFM[SD1_Inds]))-(((QFM[SD1_Inds]-RFm[SD1_Inds])/RFM[SD1_Inds])*np.log((QFM[SD1_Inds]-RFm[SD1_Inds])/RFM[SD1_Inds]))))
    Min_E[np.isnan(Min_E)] = 0
    #
    CE[np.isnan(CE)] = Min_E[np.isnan(CE)]
    return CE, Ind_E, Min_E

def ESE4(x,SD,RFm,RFM,QFm,QFM,Ts,Min_Overlap,Max_Overlap):
    """
    This function takes the observed inputs and uses the ESE4 formulation of ES to caclulate the observed Conditional Entropy (CE),
    Independent Entropy (Ind_E) and Minimum Entropy (Min_E).
    """
    G1_E = (RFm/Ts)*((-(((RFm-QFM+x)/RFm)*np.log((RFm-QFM+x)/RFm))-(((QFM-x)/RFm)*np.log((QFM-x)/RFm))))
    G2_E = (RFM/Ts)*((-(((RFM-x)/RFM)*np.log((RFM-x)/RFM))-(((x)/RFM)*np.log((x)/RFM))))
    CE = np.where(np.isnan(G1_E), 0, G1_E) + np.where(np.isnan(G2_E), 0, G2_E)
    Ind_E = (QFm/Ts)*(-np.log((QFm/Ts))) + (QFM/Ts)*(-np.log((QFM/Ts)))
    #
    Min_E = np.zeros(SD.shape[0])
    SD_1_Inds = np.where(SD == -1)[0]
    Min_E[SD_1_Inds] = (RFM[SD_1_Inds]/Ts)*((-(((RFM[SD_1_Inds]-Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds])*np.log((RFM[SD_1_Inds]-Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds]))-(((Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds])*np.log((Min_Overlap[SD_1_Inds])/RFM[SD_1_Inds]))))
    SD1_Inds = np.where(SD == 1)[0]
    Min_E[SD1_Inds] = (RFM[SD1_Inds]/Ts)*((-(((RFM[SD1_Inds]-Max_Overlap[SD1_Inds])/RFM[SD1_Inds])*np.log((RFM[SD1_Inds]-Max_Overlap[SD1_Inds])/RFM[SD1_Inds]))-(((Max_Overlap[SD1_Inds])/RFM[SD1_Inds])*np.log((Max_Overlap[SD1_Inds])/RFM[SD1_Inds]))))
    Min_E[np.isnan(Min_E)] = 0
    #
    CE[np.isnan(CE)] = Min_E[np.isnan(CE)]
    return CE, Ind_E, Min_E

##### Combinatorial cluster marker gene identification functions #####

## For any dataset, we may cluster the samples into groups. We may then be interested in which features best describe different groupings
# of the data. However, we may not know what resolution of the data best describes the patterns we are interested in. This ambuguity may
# be remedied by looking at every combination of clusters for an intentionally overclustered dataset. However, searching every combination
# of clusters quickly becomes computationally intractible.  To overcome this challange, we introduce a combinatorial clustering algorithm
# that turns the the combinatorial problem into a linear one, which can be tractably solved in mamy practical scenarios.

def Find_Max_ESSs(adata,Secondary_Features_Label):
    """
    This function takes an anndata object containing an attribute relating to a set of Secondary_Features and a attributes containing
    the ESS and SG Entropy Sorting metrics calculated pairwise for each feature Secondary_Features against each feature of the
    variables contained in the anndata object using the Parallel_Calc_ES_Matricies function. For combinatorial cluster marker 
    gene identification, Secondary_Features is created using the intentionally over-clustered samples labels of the samples in
    adata and converting them into a 2D-array through one hot encoding. This function will then identify which combination of
    one-hotted clusters maximises the correlation with the expression profile of each gene and hence identifies where in the data
    a gene may be considered a marker gene without providing any prior knowledge. The fucntion then attaches the optimum cluster
    combination and it's correspoinding ESS for each feature to the anndata object, using Secondary_Features_Label as an identifier.
    """
    ###
    print("For this function to work, you must have ran the Parallel_Calc_ES_Matricies function in a manner that attaches ESS and SG objects to your adata object that relate to your Secondary_Features_Label label.")
    ## Create the global Global_Scaled_Matrix array for faster parallel computing calculations
    global Global_Scaled_Matrix
    Global_Scaled_Matrix = adata.layers["Scaled_Counts"]   
    ### Extract the Secondary_Features object from adata
    Secondary_Features = adata.obsm[Secondary_Features_Label]
    ### Extract the Secondary_Features ESSs from adata
    All_ESSs = adata.varm[Secondary_Features_Label + "_ESSs"]
    Initial_Max_ESSs = np.asarray(np.max(All_ESSs,axis=1))
    Max_ESSs = Initial_Max_ESSs.copy()
    ### Extract the Secondary_Features SGs from adata
    All_SGs = np.asarray(adata.varm[Secondary_Features_Label + "_SGs"]).copy()
    All_SGs[All_ESSs < 0] = All_SGs[All_ESSs < 0] * -1 # For ordering we need to include the SG sort directions which we can extract from the ESSs
    ### For each feature in adata, sort the SGs of the Secondary_Features from highest to lowest. This is the main step that allows
    # use to turn the intractible combinatorial problem into a tractable linear problem.
    Sorted_SGs_Inds = np.argsort(All_SGs)[:, ::-1]
    ### Parallel warpper function
    Max_ESSs, Max_EPs, Top_Score_Columns_Combinations, Top_Score_Secondary_Features = Parallel_Identify_Max_ESSs(Secondary_Features,Sorted_SGs_Inds,Use_Cores=-1)
    ## Compile results
    Combinatorial_Label_Info = np.column_stack([np.array(Max_ESSs),np.array(Max_EPs)])#,np.array(Top_Score_Columns_Combinations,dtype="object")])
    Combinatorial_Label_Info = pd.DataFrame(Combinatorial_Label_Info,index=adata.var_names,columns=["Max_ESSs","EPs"])#,"Top_Score_Columns_Combinations"])
    ### Save Max_Combinatorial cluster information and scores
    adata.varm[Secondary_Features_Label + "_Max_Combinatorial_ESSs"] = Combinatorial_Label_Info
    print("Max combinatorial ESSs for given sample labels has been saved as 'adata.varm['" + Secondary_Features_Label + "_Max_Combinatorial_ESSs" + "']'")
    ### Save the new features/clusters that maximise the ESS scores of each feature in adata
    adata.obsm[Secondary_Features_Label + "_Max_ESS_Features"] = csc_matrix(Top_Score_Secondary_Features.astype("f"))
    print("The features/clusters relating to each Max_ESS have been saved in 'adata.obsm['" + Secondary_Features_Label + "_Max_ESS_Features" + "']'")
    return adata


def Parallel_Identify_Max_ESSs(Secondary_Features,Sorted_SGs_Inds,Use_Cores=-1):
    """
    Parallelised version of Identify_Max_ESSs function
    """
    #
    Feature_Inds = np.arange(Sorted_SGs_Inds.shape[0])
    #
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    ## Perform calculations
    print("Calculating ESS and EP matricies.")
    print("If progress bar freezes consider increasing system memory or reducing number of cores used with the 'Use_Cores' parameter as you may have hit a memory ceiling for your machine.")
    # if __name__ == '__main__':
    with np.errstate(divide='ignore',invalid='ignore'):
        Results = p_map(partial(Identify_Max_ESSs,Secondary_Features=Secondary_Features,Sorted_SGs_Inds=Sorted_SGs_Inds), Feature_Inds, num_cpus=Use_Cores)
    ## Extract results for each feature in adata
    Max_ESSs = np.zeros(len(Results)).astype("f")
    Max_EPs = np.zeros(len(Results)).astype("f")
    Top_Score_Columns_Combinations = [[]] * len(Results)
    Top_Score_Secondary_Features = np.zeros((Secondary_Features.shape[0],Feature_Inds.shape[0]))
    for i in np.arange(len(Results)):
        ESSs = Results[i][0]
        EPs = Results[i][1]
        Max_ESS_Ind = np.argmax(ESSs)
        Max_ESSs[i] = ESSs[Max_ESS_Ind]
        Max_EPs[i] = EPs[Max_ESS_Ind]
        Top_Score_Columns_Combinations[i] = Sorted_SGs_Inds[i,:][np.arange(0,Max_ESS_Ind+1)].tolist()
        Top_Score_Secondary_Features[:,i] = Secondary_Features[:,Top_Score_Columns_Combinations[i]].sum(axis=1).A.reshape(-1)
    ## Return results
    return Max_ESSs, Max_EPs, Top_Score_Columns_Combinations, Top_Score_Secondary_Features


def Identify_Max_ESSs(FF_Ind,Secondary_Features,Sorted_SGs_Inds):
    """
    For each fixed feature (FF_Ind) in adata, this function identifies which combination of Secondary_Features (the one hot clustering
    of the intentionally overclustered samples in adata) maximises the ESS of the the fixed feature, thereby giving us a coarse grain
    approximation of how to cluster the data without having to decide how many clusters we expect there to be in the data.
    """
    ## Extract the fixed feature from adata
    Fixed_Feature = Global_Scaled_Matrix[:,FF_Ind].A
    Sample_Cardinality = Fixed_Feature.shape[0]
    ## Remove the lowest rank cluster to avoid a potential cluster size being equal to the number of samples in the data.
    Sort_Order = np.delete(Sorted_SGs_Inds[FF_Ind,:],-1)
    ## From the ordered one-hut clusters, take the cumulative row sums, thereby creating the set of linearly combined one-hot
    # clusters for which we will calculate the ESSs of the fixed feature against. 
    Secondary_Features = np.cumsum(Secondary_Features.A[:, Sort_Order], axis=1)
    Secondary_Features = csc_matrix(Secondary_Features.astype("f"))
    #### Calculate constants required for ES calculations
    SF_Sums = Secondary_Features.A.sum(axis=0) 
    SF_Minority_States = SF_Sums.copy()
    SF_Minority_States[SF_Minority_States >= (Sample_Cardinality/2)] = Sample_Cardinality - SF_Minority_States[SF_Minority_States >= (Sample_Cardinality/2)]
    ##
    Fixed_Feature = Fixed_Feature.reshape(Sample_Cardinality,1) # Might be superfluous
    ## Calculate feature sums
    Fixed_Feature_Cardinality = np.sum(Fixed_Feature)
    Fixed_Feature_Minority_State = Fixed_Feature_Cardinality.copy()
    if Fixed_Feature_Minority_State >= (Sample_Cardinality/2):
        Fixed_Feature_Minority_State = Sample_Cardinality - Fixed_Feature_Minority_State
    #
    ## Identify where FF is the QF or RF
    FF_QF_Vs_RF = np.zeros(SF_Minority_States.shape[0])
    FF_QF_Vs_RF[np.where(Fixed_Feature_Minority_State > SF_Minority_States)[0]] = 1 # 1's mean FF is QF
    ## Caclulate the QFms, RFms, RFMs and QFMs for each FF and secondary feature pair
    RFms = SF_Minority_States.copy()
    Switch = np.where(FF_QF_Vs_RF == 0)[0]
    RFms[Switch] = Fixed_Feature_Minority_State
    RFMs = Sample_Cardinality - RFms
    QFms = SF_Minority_States.copy()
    Switch = np.where(FF_QF_Vs_RF == 1)[0]
    QFms[Switch] = Fixed_Feature_Minority_State
    QFMs = Sample_Cardinality - QFms
    ## Calculate the values of (x) that correspond to the maximum for each overlap scenario (mm, Mm, mM and MM) (m = minority, M = majority)
    Max_Ent_x_mm = (RFms * QFms)/(RFms + RFMs)
    Max_Ent_x_Mm = (QFMs * RFms)/(RFms + RFMs)
    Max_Ent_x_mM = (RFMs * QFms)/(RFms + RFMs)
    Max_Ent_x_MM = (RFMs * QFMs)/(RFms + RFMs)
    Max_Ent_Options = np.array([Max_Ent_x_mm,Max_Ent_x_Mm,Max_Ent_x_mM,Max_Ent_x_MM])
    ####
    # Caclulate the overlap between the FF states and the secondary features, using the correct ESE (1-4)
    All_Use_Cases, All_Overlaps_Options, All_Used_Inds = Identify_Max_ESSs_Get_Overlap_Info(Fixed_Feature,Fixed_Feature_Cardinality,Sample_Cardinality,SF_Sums,FF_QF_Vs_RF,Secondary_Features)
    ## Having extracted the overlaps and their respective ESEs (1-4), calcualte the ESS and EPs
    ESSs, D_EPs, O_EPs, SWs, SGs = Calc_ESSs(RFms, QFms, RFMs, QFMs, Max_Ent_Options, Sample_Cardinality, All_Overlaps_Options, All_Use_Cases, All_Used_Inds)
    # EPs = np.maximum(D_EPs,O_EPs)
    Identical_Features = np.where(ESSs==1)[0]
    D_EPs[Identical_Features] = 0
    O_EPs[Identical_Features] = 0
    EPs = nanmaximum(D_EPs, O_EPs)
    return ESSs, EPs


def Identify_Max_ESSs_Get_Overlap_Info(Fixed_Feature,Fixed_Feature_Cardinality,Sample_Cardinality,Feature_Sums,FF_QF_Vs_RF,Secondary_Features):
    """
    This function is an adapted version of Get_Overlap_Info where the 1D fixed feature is a variable/column from adata which is being
    compared against a set of secondary features (instead of the reverse scenario in Get_Overlap_Info). This was done because Secondary_Features
    has to be newly generated in the Identify_Max_ESSs function for each variable/column in adata.

    For any pair of features the ES mathematical framework has a set of logical rules regarding how the ES metrics
    should be calcualted. These logical rules dictate which of the two features is the reference feature (RF) or query feature
    (QF) and which of the 4 Entropy Sort Equations (ESE 1-4) should be used (Add reference to supplemental figure of 
    new manuscript when ready).
    """    
    ## Set up an array to track which of ESE equations 1-4 the recorded observed overlap relates to (row), and if it is
    # native correlation (1) or flipped anti-correlation (-1). Row 1 = mm, row 2 = Mm, row 3 = mM, row 4 = MM.
    All_Use_Cases = np.zeros((4,Feature_Sums.shape[0]))
    ## Set up an array to track the observed overlaps between the FF and the secondary features.
    All_Overlaps_Options = np.zeros((4,Feature_Sums.shape[0]))
    ## Set up a list to track the used inds/features for each ESE
    All_Used_Inds = [[]] * 4
    #
    Non_Zero_Inds = np.where(Fixed_Feature != 0)[0]
    Sub_Secondary_Features = Secondary_Features[Non_Zero_Inds,:]
    B = csc_matrix((Fixed_Feature[Non_Zero_Inds].T[0][Sub_Secondary_Features.indices], Sub_Secondary_Features.indices, Sub_Secondary_Features.indptr))
    Overlaps = Sub_Secondary_Features.minimum(B).sum(axis=0).A[0]
    #
    Inverse_Fixed_Feature = np.max(Fixed_Feature) - Fixed_Feature
    Non_Zero_Inds = np.where(Inverse_Fixed_Feature != 0)[0]
    Sub_Secondary_Features = Secondary_Features[Non_Zero_Inds,:]
    B = csc_matrix((Inverse_Fixed_Feature[Non_Zero_Inds].T[0][Sub_Secondary_Features.indices], Sub_Secondary_Features.indices, Sub_Secondary_Features.indptr))
    Inverse_Overlaps = Sub_Secondary_Features.minimum(B).sum(axis=0).A[0]
    ## If FF is observed in it's minority state, use the following 4 steps to caclulate overlaps with every other feature
    if (Fixed_Feature_Cardinality < (Sample_Cardinality / 2)):
        #######
        ## FF and other feature are minority states & FF is QF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as mm (row 1), and which are mM when the secondary feature is flipped (row 3)
        All_Use_Cases[:,Calc_Inds] = np.array([1,-1,0,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        All_Overlaps_Options[1,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        #######
        ## FF and other feature are minority states & FF is RF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as mm (row 1), and which are Mm when the secondary feature is flipped (row 2)
        All_Use_Cases[:,Calc_Inds] = np.array([1,0,-1,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[0,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        All_Overlaps_Options[2,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        #######
        ## FF is minority, other feature is majority & FF is QF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as mM (row 4), and which are mm when the secondary feature is flipped (row 1)
        All_Use_Cases[:,Calc_Inds] = np.array([0,0,1,-1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[3,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        All_Overlaps_Options[2,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        #######
        ## FF is minority, other feature is majority & FF is RF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as Mm (row 2), and which are mm when the secondary feature is flipped (row 1)
        All_Use_Cases[:,Calc_Inds] = np.array([0,1,0,-1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[3,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        All_Overlaps_Options[1,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        #
    ## If FF is observed in it's majority state, use the following 4 steps to caclulate overlaps with every other feature
    if (Fixed_Feature_Cardinality >= (Sample_Cardinality / 2)):
        #######
        ## FF is majority, other feature is minority & FF is QF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as Mm (row 2), and which are MM when the secondary feature is flipped (row 4)
        All_Use_Cases[:,Calc_Inds] = np.array([-1,1,0,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[1,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        All_Overlaps_Options[0,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        #######
        ## FF is majority, other feature is minority & FF is RF
        Calc_Inds = np.where((Feature_Sums < (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as mM (row 3), and which are MM when the secondary feature is flipped (row 4)
        All_Use_Cases[:,Calc_Inds] = np.array([-1,0,1,0]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[2,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        All_Overlaps_Options[0,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mm
        All_Used_Inds[0] = np.append(All_Used_Inds[0],Calc_Inds)
        #######
        ## FF is majority, other feature is majority & FF is QF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 1))[0]
        ## Track which features are observed as MM (row 4), and which are Mm when the secondary feature is flipped (row 2)
        All_Use_Cases[:,Calc_Inds] = np.array([0,0,-1,1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[2,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_mM
        All_Used_Inds[2] = np.append(All_Used_Inds[2],Calc_Inds)
        All_Overlaps_Options[3,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        #######
        ## FF is majority, other feature is majority & FF is RF
        Calc_Inds = np.where((Feature_Sums >= (Sample_Cardinality / 2)) & (FF_QF_Vs_RF == 0))[0]
        ## Track which features are observed as MM (row 4), and which are mM when the secondary feature is flipped (row 3)
        All_Use_Cases[:,Calc_Inds] = np.array([0,-1,0,1]).reshape(4,1)
        ## Calcualte the overlaps as the sum of minimums between samples, using Global_Scaled_Matrix for natural observations
        # and Global_Scaled_Matrix_Inverse for inverse observations.
        All_Overlaps_Options[1,Calc_Inds] = Inverse_Overlaps[Calc_Inds] # Overlaps_Mm
        All_Used_Inds[1] = np.append(All_Used_Inds[1],Calc_Inds)
        All_Overlaps_Options[3,Calc_Inds] = Overlaps[Calc_Inds] # Overlaps_MM
        All_Used_Inds[3] = np.append(All_Used_Inds[3],Calc_Inds)
        #
    return All_Use_Cases, All_Overlaps_Options, All_Used_Inds

##### Find minimal set of marker genes functions #####

def Find_Minimal_Combinatorial_Gene_Set(adata,N,Secondary_Features_Label,Input_Genes=np.array([]),Num_Reheats=5,Resolution=1,Use_Cores=-1):
    """
    Having used Find_Max_ESSs identify a set of features/clusters that maximise the ESS of each variable/column in adata and Parallel_Calc_ES_Matricies
    to calculate the ESSs of every varible/column in adata in relation to each ESS_Max feature/cluster, we can now use Find_Minimal_Combinatorial_Gene_Set
    to identify a set of N clusters that maximally capture distinct gene expression patterns in the counts matrix of adata.
    """    
    #
    Cores_Available = multiprocess.cpu_count()
    print("Cores Available: " + str(Cores_Available))
    if Use_Cores == -1:
        Use_Cores = Cores_Available - 1 # -1 Is an arbitrary buffer of idle cores that I set.
        if Use_Cores < 1:
            Use_Cores = 1
    print("Cores Used: " + str(Use_Cores))
    # Gene set optimisation
    if Input_Genes.shape[0] == 0:
        Input_Genes = np.array(adata.var_names.tolist())
    #
    Input_Gene_Inds = np.where(np.isin(adata.var_names,Input_Genes))[0]
    ###
    global Clust_ESSs
    Clust_ESSs = np.asarray(adata.varm[Secondary_Features_Label+"_ESSs"])[np.ix_(Input_Gene_Inds,Input_Gene_Inds)]
    Clust_ESSs[Clust_ESSs < 0] = 0
    ###
    Max_ESSs = np.max(Clust_ESSs,axis=1)
    ###
    Chosen_Clusts = np.random.choice(Clust_ESSs.shape[0],N,replace=False)
    ###
    Best_Score = -np.inf
    Reheat = 0
    while Reheat <= Num_Reheats:
        ###
        Chosen_Pairwise_ESSs = Clust_ESSs[np.ix_(Chosen_Clusts,Chosen_Clusts)]
        np.fill_diagonal(Chosen_Pairwise_ESSs,0)
        Current_Score = np.sum(Max_ESSs[Chosen_Clusts] - (np.max(Chosen_Pairwise_ESSs,axis=1) * Resolution))
        ###
        End = 0
        ###
        while End == 0:
            ###
            All_Max_Changes, All_Max_Change_Inds, All_Max_Replacement_Scores = All_Max_Changes, All_Max_Change_Inds, All_Max_Replacement_Scores = Parallel_Replace_Clust(N,Chosen_Pairwise_ESSs,Chosen_Clusts,Current_Score,Max_ESSs,Resolution,Use_Cores=Use_Cores)
            ###
            Replace_Clust_Ind = np.argmax(All_Max_Changes)
            if All_Max_Changes[Replace_Clust_Ind] > 0:
                # print(All_Max_Replacement_Scores[Replace_Clust_Ind])
                Replacement_Clust_Ind = All_Max_Change_Inds[Replace_Clust_Ind]
                #
                # print(Replacement_Gene_Ind)
                Chosen_Clusts[Replace_Clust_Ind] = Replacement_Clust_Ind
                #
                Chosen_Pairwise_ESSs = Clust_ESSs[np.ix_(Chosen_Clusts,Chosen_Clusts)]
                np.fill_diagonal(Chosen_Pairwise_ESSs,0)
                Current_Score = np.sum(Max_ESSs[Chosen_Clusts] - (np.max(Chosen_Pairwise_ESSs,axis=1) * Resolution))
                #
                # print(Current_Score)
                if Current_Score > Best_Score:
                    Best_Score = Current_Score
                    print("Current highest score: " + str(Best_Score))
                    Best_Chosen_Clusters = Chosen_Clusts.copy()
            else:
                End = 1
        #
        Reheat_Num = int(np.ceil(N*0.25))
        Random_Reheat_1 = np.random.choice(Input_Genes.shape[0],Reheat_Num,replace=False)
        Random_Reheat_2 = np.random.randint(Chosen_Clusts.shape[0],size=Reheat_Num)
        Chosen_Clusts[Random_Reheat_2] = Random_Reheat_1
        Reheat = Reheat + 1
        print("Reheat number: " + str(Reheat))
    #
    Chosen_Pairwise_ESSs = Clust_ESSs[np.ix_(Chosen_Clusts,Chosen_Clusts)]
    return Best_Chosen_Clusters, Input_Genes[Best_Chosen_Clusters], Chosen_Pairwise_ESSs


def Replace_Clust(Replace_Ind, Chosen_Pairwise_ESSs, Chosen_Clusts, Current_Score, Max_ESSs, Resolution):
    Sub_Chosen_Clusts = np.delete(Chosen_Clusts,Replace_Ind)
    #
    Sub_Chosen_Pairwise_ESSs = np.delete(Chosen_Pairwise_ESSs,Replace_Ind,axis=0) # Delete a row, which should be the cluster
    Sub_Chosen_Pairwise_ESSs = np.delete(Sub_Chosen_Pairwise_ESSs,Replace_Ind,axis=1)
    Sub_2nd_Maxs = np.max(Sub_Chosen_Pairwise_ESSs,axis=1)
    Replacement_Columns = Clust_ESSs[Sub_Chosen_Clusts,:]
    Replacement_Columns[(np.arange(Sub_Chosen_Clusts.shape[0]),Sub_Chosen_Clusts)] = 0
    Sub_2nd_Maxs = np.maximum(Sub_2nd_Maxs[:,np.newaxis],Replacement_Columns)
    #
    Replacement_Scores_1 = np.sum((Max_ESSs[Sub_Chosen_Clusts,np.newaxis] - (Sub_2nd_Maxs * Resolution)),axis=0)
    #
    Replacement_Rows = Clust_ESSs[:,Sub_Chosen_Clusts]
    Replacement_Rows[(Sub_Chosen_Clusts,np.arange(Sub_Chosen_Clusts.shape[0]))] = 0
    #
    Replacement_Scores_2 = Max_ESSs - (np.max(Replacement_Rows,axis=1) * Resolution)
    #
    Replacement_Scores = Replacement_Scores_1 + Replacement_Scores_2
    ###
    Changes = Replacement_Scores - Current_Score
    Changes[Chosen_Clusts] = -np.inf
    Max_Ind = np.argmax(Changes)
    Max_Change = Changes[Max_Ind]
    # All_Max_Changes[i] = Max_Change
    # All_Max_Change_Inds[i] = Max_Ind
    # All_Max_Replacement_Scores[i] = Replacement_Scores[Max_Ind]
    #
    return Max_Change, Max_Ind, Replacement_Scores[Max_Ind]

def Parallel_Replace_Clust(N,Chosen_Pairwise_ESSs,Chosen_Clusts,Current_Score,Max_ESSs,Resolution,Use_Cores):
    Replace_Inds = np.arange(N)
    #
    pool = multiprocess.Pool(processes = Use_Cores)
    Results = pool.map(partial(Replace_Clust,Resolution=Resolution,Chosen_Pairwise_ESSs=Chosen_Pairwise_ESSs,Chosen_Clusts=Chosen_Clusts,Current_Score=Current_Score,Max_ESSs=Max_ESSs), Replace_Inds)
    pool.close()
    pool.join()
    Results = np.asarray(Results)
    All_Max_Changes = Results[:,0]
    All_Max_Change_Inds = Results[:,1]
    All_Max_Replacement_Scores = Results[:,2]
    return All_Max_Changes, All_Max_Change_Inds, All_Max_Replacement_Scores


#### ESFS workflow plotting functions ####

def knn_Smooth_Gene_Expression(adata, Use_Genes, knn=30, metric='correlation', log_scale=False):
    #
    print("Calculating pairwise cell-cell distance matrix. Distance metric = " + metric + ", knn = " + str(knn))
    if issparse(adata.X) == True:
        distmat = squareform(pdist(adata[:,Use_Genes].X.A, metric))
        Smoothed_Expression = adata.X.A.copy()
    else:
        distmat = squareform(pdist(adata[:,Use_Genes].X, metric))
        Smoothed_Expression = adata.X.copy()
    neighbors = np.sort(np.argsort(distmat, axis=1)[:, 0:knn])
    #
    if log_scale == True:
        Smoothed_Expression = np.log2(Smoothed_Expression+1)
    #
    Neighbour_Expression = Smoothed_Expression[neighbors]
    Smoothed_Expression = np.mean(Neighbour_Expression,axis=1)

    print("A Smoothed_Expression sparse csc_matrix matrix with knn = " + str(knn) + " has been saved to 'adata.layers['Smoothed_Expression']'")
    adata.layers["Smoothed_Expression"] = csc_matrix(Smoothed_Expression.astype("f"))
    return adata


def ES_Rank_Genes(adata,ESS_Threshold,EP_Threshold=0,Exclude_Genes=np.array([]),Known_Important_Genes=np.array([]),Secondary_Features_Label="Self",Min_Edges=5):
    ##
    # ESSs = adata.varp['ESSs']
    Masked_ESSs = adata.varm[Secondary_Features_Label+'_ESSs'].copy()
    Masked_EPs = adata.varm[Secondary_Features_Label+'_EPs'].copy()
    Mask = np.where((Masked_EPs < EP_Threshold) | (Masked_ESSs < ESS_Threshold))
    Masked_ESSs[Mask] = 0
    Masked_EPs[Mask] = 0
    Used_Features = np.asarray(adata.var.index)
    Used_Features_Inds = np.arange(Used_Features.shape[0])
    ##
    Low_Connectivity = Used_Features[np.where(np.sum((Masked_EPs > 0),axis=0) < Min_Edges)[0]]
    Remove_Genes = np.unique(np.append(Exclude_Genes,Low_Connectivity))
    ##
    print("Pruning ESS graph by removing genes with with low numbers of edges (Min_Edges = " + str(Min_Edges) + ")")
    print("Starting genes = " + str(Used_Features_Inds.shape[0]))
    while Remove_Genes.shape[0] > 0:
        Exclude_Gene_Inds = np.where(np.isin(Used_Features,Remove_Genes))[0]
        #
        Used_Features_Inds = np.delete(Used_Features_Inds,Exclude_Gene_Inds)
        Used_Features = np.delete(Used_Features,Exclude_Gene_Inds)
        # Absolute_ESSs = Absolute_ESSs[np.ix_(Used_Features_Inds,Used_Features_Inds)]
        # ESSs = ESSs[np.ix_(Used_Features_Inds,Used_Features_Inds)]
        Masked_EPs = Masked_EPs[np.ix_(Used_Features_Inds,Used_Features_Inds)]
        Masked_ESSs = Masked_ESSs[np.ix_(Used_Features_Inds,Used_Features_Inds)]
        #
        Used_Features_Inds = np.arange(Used_Features.shape[0])
        print("Remaining genes = " + str(Used_Features_Inds.shape[0]))
        Remove_Genes = Used_Features[np.where(np.sum((Masked_EPs > 0),axis=0) < Min_Edges)[0]]
    ##
    # Masked_ESSs = Absolute_ESSs.copy()
    # Masked_ESSs[np.where((Masked_EPs < EP_Threshold) | (Absolute_ESSs < ESS_Threshold))] = 0
    ##
    print("Caclulating feature weights")
    Feature_Weights = np.average(Masked_ESSs,weights=Masked_EPs,axis=0)
    Significant_Genes_Per_Gene = (Masked_EPs > EP_Threshold).sum(1)
    Normalised_Network_Feature_Weights = Feature_Weights/Significant_Genes_Per_Gene
    ##
    if Known_Important_Genes.shape[0] > 0:
        Ranks = pd.DataFrame(np.zeros((1,Known_Important_Genes.shape[0])),index=["Rank"],columns=Known_Important_Genes)
        Rank_Sorted = Used_Features[np.argsort(-Normalised_Network_Feature_Weights)]
        for i in np.arange(Known_Important_Genes.shape[0]):
            Rank = np.where(Rank_Sorted == Known_Important_Genes[i])[0]
            if Rank.shape[0] > 0:
                Ranks[Known_Important_Genes[i]] = Rank
            else:
                Ranks[Known_Important_Genes[i]] = np.nan
        ##
        print("Known inportant gene ranks:")
        print(Ranks)
    ##
    Use_Inds = np.argsort(-Normalised_Network_Feature_Weights)
    Normalised_Weights = pd.DataFrame(Normalised_Network_Feature_Weights[Use_Inds],index=Used_Features[Use_Inds])
    Ranked_Genes = pd.DataFrame(np.arange(Use_Inds.shape[0]).astype("i"),index=Used_Features[Use_Inds])
    ##
    adata.var["ESFS_Gene_Weights"] = Normalised_Weights
    print("ESFS gene weights have been saved to 'adata.var['ESFS_Gene_Weights']'")
    adata.var["ES_Rank"] = Ranked_Genes
    print("ESFS gene ranks have been saved to 'adata.var['ES_Rank']'")
    ##
    return adata


def Plot_Top_Ranked_Genes_UMAP(adata,Top_Ranked_Genes,Clustering="None",Known_Important_Genes=np.array([]),UMAP_min_dist=0.1,UMAP_Neighbours=20,hdbscan_min_cluster_size=50,Secondary_Features_Label="Self"):
    print("Visualising the ESS graph of the top " + str(Top_Ranked_Genes) + " ranked genes in a UMAP.")
    Top_ESS_Gene_Inds = np.where(adata.var["ES_Rank"] < Top_Ranked_Genes)[0]
    Top_ESS_Genes = adata.var["ES_Rank"].index[Top_ESS_Gene_Inds]
    ##
    Masked_ESSs = adata.varm[Secondary_Features_Label+'_ESSs'].copy()[np.ix_(Top_ESS_Gene_Inds,Top_ESS_Gene_Inds)]
    # Masked_ESSs[adata.varp["EPs"][np.ix_(Top_ESS_Gene_Inds,Top_ESS_Gene_Inds)] < 0] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Gene_Embedding = umap.UMAP(n_neighbors=UMAP_Neighbours, min_dist=UMAP_min_dist, n_components=2, random_state=42).fit_transform(Masked_ESSs)
        # Gene_Embedding = umap.UMAP(n_neighbors=UMAP_Neighbours, min_dist=UMAP_min_dist, n_components=2).fit_transform(Masked_ESSs)
    ##
    # No clustering
    if Clustering == "None":
        print("Clustering == 'None', set Clustering to numeric value for Kmeans clustering or 'hdbscan' for automated density clustering.")
        plt.figure(figsize=(5,5))
        plt.title("Top " + str(Top_Ranked_Genes) + " genes UMAP", fontsize=20)
        plt.scatter(Gene_Embedding[:,0],Gene_Embedding[:,1],s=7)
        plt.xlabel("UMAP 1",fontsize=16)
        plt.ylabel("UMAP 2",fontsize=16)
        Labels = np.zeros(Top_Ranked_Genes)
    # Kmeans clustering
    if isinstance(Clustering,int):
        print("Clustering == an integer value leading to Kmeans clustering, set Clustering to 'hdbscan' for automated density clustering.")
        kmeans = KMeans(n_clusters=Clustering, random_state=42, n_init="auto").fit(Gene_Embedding)
        Labels = kmeans.labels_
        Unique_Labels = np.unique(Labels)
        #
        plt.figure(figsize=(5,5))
        plt.title("Top " + str(Top_Ranked_Genes) + " genes UMAP\nClustering = Kmeans", fontsize=20)
        for i in np.arange(Unique_Labels.shape[0]):
            IDs = np.where(Labels == Unique_Labels[i])[0]
            plt.scatter(Gene_Embedding[IDs,0],Gene_Embedding[IDs,1],s=7,label=Unique_Labels[i])
        #
        plt.xlabel("UMAP 1",fontsize=16)
        plt.ylabel("UMAP 2",fontsize=16)
    # hdbscan clustering
    if Clustering == "hdbscan":
        print("Clustering == 'hdbscan', set Clustering to an integer value for automated Kmeans clustering.")
        hdb = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        np.random.seed(42)
        hdb.fit(Gene_Embedding)
        Labels = hdb.labels_
        Unique_Labels = np.unique(Labels)
        #
        plt.figure(figsize=(5,5))
        plt.title("Top " + str(Top_Ranked_Genes) + " genes UMAP\nClustering = hdbscan", fontsize=20)
        for i in np.arange(Unique_Labels.shape[0]):
            IDs = np.where(Labels == Unique_Labels[i])[0]
            plt.scatter(Gene_Embedding[IDs,0],Gene_Embedding[IDs,1],s=7,label=Unique_Labels[i])
        #
        plt.xlabel("UMAP 1",fontsize=16)
        plt.ylabel("UMAP 2",fontsize=16)
    #
    if Known_Important_Genes.shape[0] > 0:
        Important_Gene_Inds = np.where(np.isin(Top_ESS_Genes,Known_Important_Genes))[0]
        plt.scatter(Gene_Embedding[Important_Gene_Inds,0],Gene_Embedding[Important_Gene_Inds,1],s=15,c="black",marker="x",label="Known important genes")
    #
    plt.legend()
    print("This function outputs the 'Top_ESS_Genes', 'Gene_Clust_Labels' and 'Gene_Embedding' objects in case users would like to investiage them further.")
    return Top_ESS_Genes, Labels, Gene_Embedding


def Get_Gene_Cluster_Cell_UMAPs(adata,Gene_Clust_Labels,Top_ESS_Genes,n_neighbors,min_dist,log_transformed,specific_cluster=None):
    print("Generating the cell UMAP embeddings for each cluster of genes from the previous function.")
    if specific_cluster == None:
        Unique_Gene_Clust_Labels = np.unique(Gene_Clust_Labels)
        Gene_Cluster_Selected_Genes = [[]] * Unique_Gene_Clust_Labels.shape[0]
        Gene_Cluster_Embeddings = [[]] * Unique_Gene_Clust_Labels.shape[0]
    else:
        Unique_Gene_Clust_Labels = np.array([specific_cluster])
        Gene_Cluster_Selected_Genes = [[]] * Unique_Gene_Clust_Labels.shape[0]
        Gene_Cluster_Embeddings = [[]] * Unique_Gene_Clust_Labels.shape[0]
    for i in np.arange(Unique_Gene_Clust_Labels.shape[0]):
        print("Plotting cell UMAP using gene clusters " + str(Unique_Gene_Clust_Labels[i]))
        Selected_Genes = np.asarray(Top_ESS_Genes[np.where(np.isin(Gene_Clust_Labels,Unique_Gene_Clust_Labels[i]))[0]])
        Gene_Cluster_Selected_Genes[i] = Selected_Genes
        # np.save(path + "Saved_ESFS_Genes.npy",np.asarray(Selected_Genes))
        Reduced_Input_Data = adata[:,Selected_Genes].X.A
        if log_transformed == True:
            Reduced_Input_Data = np.log2(Reduced_Input_Data+1)
        #
        # Embedding_Model = umap.UMAP(n_neighbors=n_neighbors, metric="correlation",min_dist=min_dist,n_components=2,random_state=42).fit(Reduced_Input_Data)
        Embedding_Model = umap.UMAP(n_neighbors=n_neighbors, metric="correlation",min_dist=min_dist,n_components=2).fit(Reduced_Input_Data)
        Embedding = Embedding_Model.embedding_
        Gene_Cluster_Embeddings[i] = Embedding
        #
    return Gene_Cluster_Embeddings, Gene_Cluster_Selected_Genes

def Plot_Gene_Cluster_Cell_UMAPs(adata, Gene_Cluster_Embeddings, Gene_Cluster_Selected_Genes, Cell_Label="None", ncol=1,log2_Gene_Expression=True):
    #
    if np.isin(Cell_Label,adata.obs.columns):
        Cell_Labels = adata.obs[Cell_Label]
        Unique_Cell_Labels = np.unique(Cell_Labels)
        #
        for i in np.arange(len(Gene_Cluster_Embeddings)):
            Embedding = Gene_Cluster_Embeddings[i]
            plt.figure(figsize=(7, 5))
            plt.title("Cell UMAP" + "\n" +
                    str(Gene_Cluster_Selected_Genes[i].shape[0]) + " genes", fontsize=20)
            for j in np.arange(Unique_Cell_Labels.shape[0]):
                IDs = np.where(Cell_Labels == Unique_Cell_Labels[j])
                plt.scatter(Embedding[IDs, 0], Embedding[IDs, 1], s=3, label=Unique_Cell_Labels[j])
            #
            plt.xlabel("UMAP 1", fontsize=16)
            plt.ylabel("UMAP 2", fontsize=16)
            #
            if Cell_Label != "None":
                # Adjust legend to be outside and below the plot
                plt.legend(
                    loc='center left',  # Align legend to the left of the bounding box
                    bbox_to_anchor=(1, 0.5),  # Position to the right and vertically centered
                    ncol=1,  # Keep in a single column to span height
                    fontsize=10,
                    frameon=False,
                    markerscale=5
                )
    #
    if np.isin(Cell_Label,adata.var.index):
        #
        Expression = adata[:,Cell_Label].X.T
        if issparse(Expression):
            Expression = Expression.todense()
        #
        Expression = np.asarray(Expression)[0]
        #
        if log2_Gene_Expression == True:
            Expression = np.log2(Expression+1)
        #
        for i in np.arange(len(Gene_Cluster_Embeddings)):
            Embedding = Gene_Cluster_Embeddings[i]
            plt.figure(figsize=(7, 5))
            plt.title("Cell UMAP" + "\n" +
                    Cell_Label, fontsize=20)
            #
            plt.scatter(Embedding[:, 0], Embedding[:, 1], s=3, c=Expression, cmap="seismic")
            #
            plt.xlabel("UMAP 1", fontsize=16)
            plt.ylabel("UMAP 2", fontsize=16)
            cb = plt.colorbar()
            cb.set_label('$log_2$(Expression)', labelpad=-50,fontsize=10)
    #
    if ((np.isin(Cell_Label,adata.obs.columns)) & (np.isin(Cell_Label,adata.var.index)) == False):
        print("Cell label or gene not found in 'adata.obs.columns' or 'adata.var.index'")

# def Convert_To_Ranked_Gene_List(adata,sample_labels):
#     ESSs = adata.varm[sample_labels+'_ESSs']
#     Columns = np.asarray(ESSs.columns)
#     Ranked_Gene_List = pd.DataFrame(np.zeros(ESSs.shape),columns=Columns)
#     for i in np.arange(Columns.shape[0]):
#         Ranked_Gene_List[Columns[i]] = ESSs.index[np.argsort(-ESSs[Columns[i]])]
#     #
#     return Ranked_Gene_List

# def Display_Chosen_Genes_gif(Embedding,Chosen_Genes,adata):
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
#         scatter = ax.scatter(Embedding[Order, 0], Embedding[Order, 1], c=Exp[Order], s=2, vmax=Vmax, cmap="seismic")
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

