# Understanding ESFS

This page explains the reasoning behind each ESFS algorithm — not just what it does, but *why* it works the way it does. If you are new to ESFS, reading this alongside the [example workflows](getting_started.md) will help you get the most out of your data.

---

## Why gene expression space?

Most scRNA-seq analysis workflows apply dimensionality reduction (PCA) or batch integration before clustering and visualisation. These steps compress thousands of gene dimensions into a much smaller latent space, which makes the data easier to work with — but at a cost.

**The problem with latent representations:**
- Compression necessarily discards variation. If that variation is biological rather than technical, it is lost.
- Batch integration algorithms are designed to remove inter-dataset differences, but biological differences between conditions or time points can be inadvertently removed alongside technical noise.
- The result is a latent space that may look clean but can distort cell–cell relationships and obscure meaningful expression dynamics.

**What ESFS does instead:**

ESFS identifies a subset of genes in which biological signal consistently outweighs technical noise. Once those genes are found, the raw expression matrix can be used directly to build high-resolution embeddings — no PCA, no batch correction required. Because you are working in the original gene expression space, every dimension has a direct biological interpretation, and downstream results (marker genes, trajectories, spatial patterns) are straightforwardly traceable back to the data.

---

## ES-GSS: beyond highly variable genes

**The problem with HVG selection:**

Highly variable gene (HVG) selection ranks genes by how much their expression varies across cells — a univariate measure. A gene can be highly variable because it is a genuine biological signal, or because it is noisy, sparsely expressed, or affected by technical confounders. HVG selection cannot distinguish between these cases, which is why it can inadvertently include uninformative genes and exclude biologically meaningful ones.

**How ES-GSS works:**

ES-GSS takes a pairwise approach. For every pair of genes, it computes an Entropy Sort Score (ESS) — an information-theoretic measure of how strongly the expression of one gene predicts the ordering of another. Genes that are strongly and consistently predictive of many other genes are ranked highly. Genes whose variation is idiosyncratic or noisy tend to have weak, inconsistent ESS values and rank poorly.

The result is a weighted gene network, where edge weights reflect pairwise information content. Highly connected genes — those that co-vary with many others in a structured way — are the most informative of biological cell state. These are the genes ES-GSS selects.

**In practice:** after running ES-GSS, genes cluster into modules by co-expression pattern. You select the module whose cell UMAP embedding best captures the biological structure you are interested in (distinct cell types, developmental trajectories, spatial gradients, etc.). See the [parameter guide](getting_started.md#parameter-guide) for advice on choosing `Num_Top_Ranked_Genes`.

---

## ES-CCF: why intentional over-clustering?

**The problem with fixed clustering resolution:**

When you cluster scRNA-seq data at a single resolution, you implicitly assume that one partitioning of the data is appropriate for all genes. In reality, different genes mark populations at different scales: some are exclusive to a single fine-grained cluster, others are shared across broad groups of related cells. A low resolution misses rare populations; a high resolution over-partitions cells into artificial groups. No single resolution is optimal for every gene simultaneously.

**How ES-CCF works:**

ES-CCF sidesteps this problem entirely by first over-clustering the data at a high resolution (deliberately generating more clusters than you expect to be biologically meaningful), then — for each gene — searching for the *combination* of those fine-grained clusters that maximises the correlation between the gene's expression and the cluster membership profile.

This is a combinatorial optimisation problem that would be computationally intractable if solved by brute force. ES-CCF uses the Sort Gain (SG) metric — a directional measure of how much each additional cluster improves the correlation — to reduce the search to a linear-time procedure solvable in minutes on a laptop.

The output for each gene is its **optimal cluster combination**: the set of cells in which that gene is most specifically and robustly expressed. This combination may span a single cluster (a specific rare population), several adjacent clusters (a broad cell type), or a set of non-adjacent clusters (a shared transcriptional program expressed across multiple lineages).

**In practice:** you do not need to choose a clustering resolution that is "correct" — you intentionally choose one that is too fine-grained (e.g. Leiden resolution 8–10), then let ES-CCF find the biologically meaningful groupings for you.

---

## ES-FMG: reading your marker gene results

**The problem with conventional marker gene selection:**

Standard differential expression analysis identifies genes that are enriched in one cluster versus all others. This has two limitations:
1. It is sensitive to the chosen clustering resolution — different resolutions give different markers.
2. It identifies genes enriched in *one* group, missing genes that mark combinations of groups (e.g. a gene expressed in two related but distinct cell types).

**How ES-FMG works:**

ES-FMG takes the per-gene optimal cluster combinations from ES-CCF and uses a simulated annealing optimisation to select a set of N genes that are simultaneously:
- **High ESS** — each gene is strongly correlated with its optimal cluster combination (it is a robust marker)
- **Mutually non-redundant** — the selected genes capture distinct populations, minimising overlap between their expression profiles

The result is a compact, interpretable set of marker genes — each associated with a specific cell population or expression program — that together span the biological diversity in your dataset.

**Reading the results:**

- Each ES-FMG gene is associated with a cluster combination (its optimal cell population). Inspect these combinations on your UMAP to understand what population each gene marks.
- The ranked gene list for each ES-FMG gene (sorted by ESS within the same cluster combination) reveals co-expressed genes in the same population — useful for identifying additional markers or understanding the biology of that population.
- Genes that mark overlapping populations (e.g. a shared transcriptional program) will appear in ES-FMG when `resolution` is set lower; genes marking exclusively distinct populations dominate at `resolution = 1`.

See the [parameter guide](getting_started.md#parameter-guide) for guidance on tuning N and `resolution`.
