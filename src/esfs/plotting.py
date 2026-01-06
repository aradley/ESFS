from typing import Optional
import warnings

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, HDBSCAN
import umap

from .backend import backend
from .ESFS import _convert_sparse_array, move_to_gpu, convert_to_numpy

xp = backend.xp
xpsparse = backend.xpsparse
USING_GPU = backend.using_gpu


def knn_smooth_gene_expression(
    adata,
    use_genes,
    knn: int = 30,
    metric: str = "correlation",
    log_scale: bool = False,
    batch_size: int = 1000,
    n_jobs: int = -1,
):
    assert metric == "correlation", "Currently only 'correlation' metric is supported."

    use_gene_idxs = xp.nonzero(xp.isin(xp.asarray(adata.var_names), use_genes))[0]

    # NOTE: If needed in future, could try to keep sparse, similar to create_scaled_matrix
    X_subset = adata[:, use_gene_idxs].X
    if xpsparse.issparse(X_subset):
        X_subset = X_subset.toarray()
    if log_scale:
        X_subset = xp.log2(X_subset + 1)

    n_cells = X_subset.shape[0]
    print(
        f"Computing batched correlation distances for {n_cells} cells, batch size = {batch_size}"
    )

    # Simplify by ensuring remainder is done on CPU
    if USING_GPU:
        X_subset = X_subset.get()

    # Function to compute distances for a batch
    def compute_batch(i_start, i_end):
        batch = X_subset[i_start:i_end]
        dists = cdist(batch, X_subset, metric=metric)
        # For each row, get top-k indices (excluding self optionally)
        neighbors_batch = np.argpartition(dists, kth=knn, axis=1)[:, :knn]
        return neighbors_batch

    # Launch batches in parallel
    batch_ranges = [
        (i, min(i + batch_size, n_cells)) for i in range(0, n_cells, batch_size)
    ]
    all_neighbors = Parallel(n_jobs=n_jobs)(
        delayed(compute_batch)(i_start, i_end) for (i_start, i_end) in batch_ranges
    )

    # Concatenate all neighbor indices
    neighbors = np.vstack(all_neighbors)

    # Get full expression matrix for smoothing
    full_X = adata.X
    if xpsparse.issparse(full_X):
        full_X = full_X.toarray()
        if USING_GPU:
            full_X = full_X.get()
    if log_scale:
        full_X = np.log2(full_X + 1)

    print(f"Smoothing expression matrix using mean over {knn} neighbors...")

    smoothed_expression = np.array(
        [np.mean(full_X[neighbor_idx], axis=0) for neighbor_idx in neighbors]
    )
    adata.layers["Smoothed_Expression"] = _convert_sparse_array(
        smoothed_expression.astype(np.float32), to_scipy=True
    )
    return adata


def ES_rank_genes(
    adata,
    EP_threshold: float = 0.0,
    ESS_threshold: float = 0.01,
    exclude_genes: Optional[tuple] = None,
    known_important_genes: Optional[tuple] = None,
    secondary_features_label: str = "Self",
    min_edges: int = 5,
):
    ##
    # ESSs = adata.varp['ESSs']
    masked_ESSs = move_to_gpu(adata.varm[secondary_features_label + "_ESSs"].copy())
    masked_EPs = move_to_gpu(adata.varm[secondary_features_label + "_EPs"].copy())
    mask = (masked_EPs < EP_threshold) | (masked_ESSs < ESS_threshold)
    masked_ESSs[mask] = 0
    masked_EPs[mask] = 0
    # used_features = xp.asarray(adata.var.index)
    # used_features_idxs = xp.arange(used_features.shape[0])
    used_features_d_idxs = {i: j for i, j in enumerate(adata.var.index)}
    used_features_d_names = {v: k for k, v in used_features_d_idxs.items()}
    ##
    low_conn_idxs = xp.nonzero(xp.sum((masked_EPs > 0), axis=0) < min_edges)[0]
    low_connectivity = [used_features_d_idxs[int(i)] for i in low_conn_idxs]
    if exclude_genes is not None:
        remove_genes = list(set(exclude_genes) | set(low_connectivity))
    else:
        remove_genes = low_connectivity
    ##
    print(
        "Pruning ESS graph by removing genes with with low numbers of edges (min_edges = "
        + str(min_edges)
        + ")"
    )
    print(f"Starting genes = {len(used_features_d_idxs)}")
    while len(remove_genes) > 0:
        for name in remove_genes:
            i = used_features_d_names.pop(name)
            used_features_d_idxs.pop(i, None)
        # absolute_ESSs = absolute_ESSs[xp.ix_(used_features_idxs,used_features_idxs)]
        # ESSs = ESSs[xp.ix_(used_features_idxs,used_features_idxs)]
        masked_EPs = masked_EPs[
            xp.ix_(list(used_features_d_idxs.keys()), list(used_features_d_idxs.keys()))
        ]
        masked_ESSs = masked_ESSs[
            xp.ix_(list(used_features_d_idxs.keys()), list(used_features_d_idxs.keys()))
        ]
        used_features_d_idxs = {
            i: j for i, j in enumerate(used_features_d_names.keys())
        }
        used_features_d_names = {v: k for k, v in used_features_d_idxs.items()}
        print(f"Remaining genes = {len(used_features_d_idxs)}")
        remove_idxs = xp.nonzero(xp.sum((masked_EPs > 0), axis=0) < min_edges)[0]
        remove_genes = [used_features_d_idxs[int(i)] for i in remove_idxs]
    ##
    # masked_ESSs = absolute_ESSs.copy()
    # masked_ESSs[xp.where((masked_EPs < EP_threshold) | (absolute_ESSs < ESS_threshold))] = 0
    ##
    print("Calculating feature weights")
    feature_weights = xp.average(masked_ESSs, weights=masked_EPs, axis=0)
    # Add check for no remaining genes to avoid inserting empty arrays
    if feature_weights.size == 0:
        raise ValueError(
            "No genes remain after pruning. Consider changing threshold or min_edges parameters."
        )
    sig_genes_per_gene = (masked_EPs > EP_threshold).sum(1)
    norm_network_feature_weights = feature_weights / sig_genes_per_gene
    ##
    sorted_indices = xp.argsort(-norm_network_feature_weights)
    if known_important_genes.shape[0] > 0:
        # print(used_features == list(used_features_d_names.keys()))
        # Get sorted indices by descending norm_network_feature_weights
        # Map sorted indices to gene names using used_features_idxs
        rank_sorted_names = [used_features_d_idxs[int(i)] for i in sorted_indices]
        gene_rank_dict = {gene: rank for rank, gene in enumerate(rank_sorted_names)}
        # Build dictionary of known important gene ranks
        df_ranks = {}
        for gene in known_important_genes:
            df_ranks[gene] = gene_rank_dict.get(gene, np.nan)
        df_ranks = pd.DataFrame(df_ranks, index=["Rank"])
        ##
        print("Known inportant gene ranks:")
        print(df_ranks)
    ##
    norm_network_feature_weights = convert_to_numpy(norm_network_feature_weights)
    norm_weights = pd.DataFrame(
        norm_network_feature_weights[sorted_indices], index=rank_sorted_names
    )
    ranked_genes = pd.DataFrame(range(sorted_indices.shape[0]), index=rank_sorted_names)
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
    clustering: str | int | None = None,
    known_important_genes=np.array([]),
    UMAP_min_dist: float = 0.1,
    UMAP_neighbours: int = 20,
    hdbscan_min_cluster_size: int = 50,
    secondary_features_label: str = "Self",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(
        "Visualising the ESS graph of the top "
        + str(top_ranked_genes)
        + " ranked genes in a UMAP."
    )
    top_ESS_gene_idxs = np.where(adata.var["ES_Rank"] < top_ranked_genes)[0]
    top_ESS_genes = adata.var["ES_Rank"].index[top_ESS_gene_idxs]
    ##
    masked_ESSs = adata.varm[secondary_features_label + "_ESSs"][
        np.ix_(top_ESS_gene_idxs, top_ESS_gene_idxs)
    ].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gene_embedding = umap.UMAP(
            n_neighbors=UMAP_neighbours,
            min_dist=UMAP_min_dist,
            n_components=2,
            random_state=random_state,
        ).fit_transform(masked_ESSs)
    ##
    # No clustering
    if clustering is None or clustering == "None":
        print(
            "Clustering == 'None', set clustering to numeric value for KMeans clustering or to 'hdbscan' for automated density clustering."
        )
        plt.figure(figsize=(5, 5))
        plt.title("Top " + str(top_ranked_genes) + " genes UMAP", fontsize=20)
        plt.scatter(gene_embedding[:, 0], gene_embedding[:, 1], s=7)
        plt.xlabel("UMAP 1", fontsize=16)
        plt.ylabel("UMAP 2", fontsize=16)
        labels = np.zeros(top_ranked_genes)
    # Kmeans clustering
    elif isinstance(clustering, int):
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
    elif clustering == "hdbscan":
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
    else:
        raise ValueError(
            "Clustering must be None/'None', an integer value for KMeans clustering, or 'hdbscan' for automated density clustering."
        )
    #
    if known_important_genes.shape[0] > 0:
        important_gene_idxs = top_ESS_genes.get_indexer(known_important_genes)
        # NOTE: If known_important_genes are not in top_ESS_genes, important_gene_idxs will be -1
        plt.scatter(
            gene_embedding[important_gene_idxs, 0],
            gene_embedding[important_gene_idxs, 1],
            s=15,
            c="black",
            marker="x",
            label="Known important genes",
        )
    #
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    print(
        "This function outputs the 'Top_ESS_Genes', 'Gene_Clust_Labels' and 'Gene_Embedding' objects in case users would like to investigate them further."
    )
    return top_ESS_genes.values, labels, gene_embedding


def get_gene_cluster_cell_UMAPs(
    adata,
    gene_clust_labels: np.ndarray,
    top_ESS_genes: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    log_transformed: bool,
    specific_cluster: Optional[int] = None,
    metric: str = "correlation",
    random_state: Optional[int] = None,
    **kwargs,
):
    print(
        "Generating the cell UMAP embeddings for each cluster of genes from the previous function."
    )
    if specific_cluster is None:
        unique_gene_clust_labels = np.unique(gene_clust_labels)
    else:
        unique_gene_clust_labels = [specific_cluster]
    # Create containers for the selected genes and embeddings
    gene_cluster_selected_genes = []
    gene_cluster_embeddings = []

    for lbl in unique_gene_clust_labels:
        print(f"Plotting cell UMAP using gene cluster {lbl}")
        selected_genes = top_ESS_genes[gene_clust_labels == lbl].tolist()
        if len(selected_genes) == 0:
            print(f"No genes found in cluster {lbl}, skipping this cluster.")
            # Add to maintain the list length
            gene_cluster_embeddings.append(None)
            gene_cluster_selected_genes.append([])
            continue
        gene_cluster_selected_genes.append(selected_genes)
        # xp.save(path + "Saved_ESFS_Genes.npy",xp.asarray(selected_genes))
        reduced_input_data = adata[:, selected_genes].X.A.copy()
        if log_transformed:
            reduced_input_data = np.log2(reduced_input_data + 1)
        #
        embedding_model = umap.UMAP(
            n_neighbors=n_neighbors,
            metric=metric,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,  # UMAP uses None as default
            **kwargs,
        ).fit(reduced_input_data)
        gene_cluster_embeddings.append(embedding_model.embedding_)
        #
    return gene_cluster_embeddings, gene_cluster_selected_genes


def plot_gene_cluster_cell_UMAPs(
    adata,
    gene_cluster_embeddings: list,
    gene_cluster_selected_genes: list,
    cell_label="None",
    ncol=1,
    log2_gene_expression: bool = True,
    figsize: tuple = (18, 10),
    marker_size: int = 3,
):
    num_plots = len(gene_cluster_embeddings)
    nrow = int(np.ceil(num_plots / ncol))
    # Check if cell_label is in adata.obs or adata.var
    if cell_label in adata.obs.columns:
        cell_labels = adata.obs[cell_label]
        unique_cell_labels = np.unique(cell_labels)
        #
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
        fig.suptitle(f"Cell UMAPs by '{cell_label}'", fontsize=20)

        for idx, embedding in enumerate(gene_cluster_embeddings):
            row, col = divmod(idx, ncol)
            ax = axes[row, col]
            ax.set_title(
                f"Gene Cluster {idx} ({len(gene_cluster_selected_genes[idx])} genes)",
                fontsize=14,
            )
            for label in unique_cell_labels:
                label_idxs = np.where(cell_labels == label)
                ax.scatter(
                    embedding[label_idxs, 0],
                    embedding[label_idxs, 1],
                    s=marker_size,
                    label=label,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP 1", fontsize=12)
            ax.set_ylabel("UMAP 2", fontsize=12)
        # Move legend outside the last axis
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10
        )
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    elif cell_label in adata.var.index:
        expression = adata[:, cell_label].X.T
        if xpsparse.issparse(expression):
            expression = expression.toarray()
            if USING_GPU:
                expression = expression.get()
        expression = np.asarray(expression)[0]
        if log2_gene_expression:
            expression = np.log2(expression + 1)

        # Set up the overall grid: one extra column for the colorbar
        fig = plt.figure(figsize=figsize)
        outer_grid = gridspec.GridSpec(
            nrow, ncol + 1, width_ratios=[1] * ncol + [0.05], wspace=0.3
        )
        fig.suptitle(f"Cell UMAPs colored by '{cell_label}' expression", fontsize=20)

        for idx, Embedding in enumerate(gene_cluster_embeddings):
            row, col = divmod(idx, ncol)
            ax = fig.add_subplot(outer_grid[row, col])
            Order = np.argsort(expression)
            sc = ax.scatter(
                Embedding[Order, 0],
                Embedding[Order, 1],
                s=marker_size,
                c=expression[Order],
                cmap="seismic",
            )
            ax.set_title(
                f"Gene Cluster {idx} ({len(gene_cluster_selected_genes[idx])} genes)",
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP 1", fontsize=12)
            ax.set_ylabel("UMAP 2", fontsize=12)

        # Add one colorbar on the rightmost column
        cax = fig.add_subplot(outer_grid[:, -1])
        cb = fig.colorbar(sc, cax=cax)
        if log2_gene_expression:
            cb.set_label("$log_2$(Expression)", fontsize=10)
        else:
            cb.set_label("Expression", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        print(
            "Cell label or gene not found in 'adata.obs.columns' or 'adata.var.index'"
        )


# def Convert_To_Ranked_Gene_List(adata,sample_labels):
#     ESSs = adata.varm[sample_labels+'_ESSs']
#     Columns = xp.asarray(ESSs.columns)
#     Ranked_Gene_List = pd.DataFrame(xp.zeros(ESSs.shape),columns=Columns)
#     for i in xp.arange(Columns.shape[0]):
#         Ranked_Gene_List[Columns[i]] = ESSs.index[xp.argsort(-ESSs[Columns[i]])]
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
#         Order = xp.argsort(Exp)
#         #
#         ax.set_title(Gene, fontsize=22)
#         Vmax = xp.percentile(Exp, 99)
#         if Vmax == 0:
#             Vmax = xp.max(Exp)
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
