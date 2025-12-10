from typing import Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, HDBSCAN
import umap

from .backend import backend

xp = backend.xp
xpsparse = backend.xpsparse
USING_GPU = backend.using_gpu


def knn_Smooth_Gene_Expression(
    adata, use_genes, knn=30, metric="correlation", log_scale: bool = False
):
    #
    print(
        "Calculating pairwise cell-cell distance matrix. Distance metric = "
        + metric
        + ", knn = "
        + str(knn)
    )
    if xpsparse.issparse(adata.X):
        distmat = squareform(pdist(adata[:, use_genes].X.A, metric))
        smoothed_expression = adata.X.A.copy()
    else:
        distmat = squareform(pdist(adata[:, use_genes].X, metric))
        smoothed_expression = adata.X.copy()
    neighbors = xp.sort(xp.argsort(distmat, axis=1)[:, 0:knn])
    #
    if log_scale:
        smoothed_expression = xp.log2(smoothed_expression + 1)
    #
    neighbour_expression = smoothed_expression[neighbors]
    smoothed_expression = xp.mean(neighbour_expression, axis=1)

    print(
        "A Smoothed_Expression sparse csc_matrix matrix with knn = "
        + str(knn)
        + " has been saved to 'adata.layers['Smoothed_Expression']'"
    )
    adata.layers["Smoothed_Expression"] = xpsparse.csc_matrix(
        smoothed_expression.astype(xp.float32)
    )
    return adata


def ES_Rank_Genes(
    adata,
    ESS_threshold,
    EP_threshold=0,
    exclude_genes: Optional[tuple] = None,
    known_important_genes: Optional[tuple] = None,
    secondary_features_label="Self",
    min_edges=5,
):
    ##
    # ESSs = adata.varp['ESSs']
    masked_ESSs = adata.varm[secondary_features_label + "_ESSs"].copy()
    masked_EPs = adata.varm[secondary_features_label + "_EPs"].copy()
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
    if USING_GPU:
        norm_weights = pd.DataFrame(
            norm_network_feature_weights[sorted_indices].get(), index=rank_sorted_names
        )
    else:
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
    if USING_GPU:
        masked_ESSs = adata.varm[secondary_features_label + "_ESSs"].get()[
            np.ix_(top_ESS_gene_idxs, top_ESS_gene_idxs)
        ].copy()
    else:
        masked_ESSs = adata.varm[secondary_features_label + "_ESSs"][
            xp.ix_(top_ESS_gene_idxs, top_ESS_gene_idxs)
        ].copy()
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
    plt.legend()
    print(
        "This function outputs the 'Top_ESS_Genes', 'Gene_Clust_Labels' and 'Gene_Embedding' objects in case users would like to investiage them further."
    )
    return top_ESS_genes, labels, gene_embedding


def get_gene_cluster_cell_UMAPs(
    adata,
    gene_clust_labels,
    top_ESS_genes,
    n_neighbors: int,
    min_dist: float,
    log_transformed: bool,
    specific_cluster: Optional[int] = None,
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
        reduced_input_data = adata[:, selected_genes].X.A
        if log_transformed:
            reduced_input_data = np.log2(reduced_input_data + 1)
        #
        # embedding_model = umap.UMAP(n_neighbors=n_neighbors, metric="correlation",min_dist=min_dist,n_components=2,random_state=42).fit(reduced_input_data)
        embedding_model = umap.UMAP(
            n_neighbors=n_neighbors,
            metric="correlation",
            min_dist=min_dist,
            n_components=2,
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
):
    #
    # TODO: Refactor this to move away from strings as we have done prev with `get_gene_cluster_cell_UMAPs`
    if cell_label in adata.obs.columns:
        cell_labels = adata.obs[cell_label]
        unique_cell_labels = np.unique(cell_labels)
        #
        for i, embedding in enumerate(gene_cluster_embeddings):
            plt.figure(figsize=(7, 5))
            plt.title(
                "Cell UMAP"
                + "\n"
                + str(len(gene_cluster_selected_genes[i]))
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
        if xpsparse.issparse(expression):
            expression = expression.todense()
        #
        expression = np.asarray(expression)[0]
        #
        if log2_gene_expression:
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
    if not (np.isin(cell_label, adata.obs.columns)) & (
        np.isin(cell_label, adata.var.index)
    ):
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
