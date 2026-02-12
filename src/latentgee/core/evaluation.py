
def permanova_r2(
    X: np.ndarray,
    grouping,
    metric:str = "braycurtis",
    ids=None,
    permutations:int = 999,
    pseudocount:float = 1e-6,
):
    """
    Calculate PERMANOVA (pseudo-F, p-value) and R² like vegan::adonis2.
    
    Parameters
    ----------
    X : (n_samples, n_features) array_like
        Feature (count/proportion) matrix.
    grouping : 1-D array_like
        Group labels, length == n_samples.
    metric : str, default 'braycurtis'
        Distance metric for `pdist`.  
        Special case: 'aitchison' ⇒ clr-Euclidean.
    ids : list[str] or None
        Sample IDs for the distance matrix.
    permutations : int
        Number of permutations for PERMANOVA
    pseudocount : float
        Added to zeros before clr, if metric == 'aitchison'.
    """
    X = np.asarray(X.astype(float))
    grouping = np.asarray(grouping)

    # ---------- sample IDs ----------
    if ids is None:
        ids = [f"S{i+1}" for i in range(X.shape[0])]

    # ---------- distance matrix ----------
    if metric.lower() == "aitchison":
        # 1) zero replacement → clr transform
        X_clr = clr(multiplicative_replacement(X, pseudocount))
        # 2) Euclidean distance in clr space == Aitchison distance
        dist = squareform(pdist(X_clr, metric="euclidean"))
    else:
        dist = squareform(pdist(X, metric=metric))

    dm = DistanceMatrix(dist, ids=ids)

    # ---------- R² ----------
    d2 = dm.data ** 2
    n = d2.shape[0]
    sst = d2[np.triu_indices(n, 1)].sum() / n

    ssw = 0.0
    for g in np.unique(grouping):
        idx = np.where(grouping == g)[0]
        if len(idx) < 2:
            continue
        d2_g = d2[np.ix_(idx, idx)]
        ssw += d2_g[np.triu_indices(len(idx), 1)].sum() / len(idx)

    ssb = sst - ssw
    r2 = ssb / sst

    # ---------- PERMANOVA ----------
    res = permanova(dm, grouping, permutations=permutations)

    return res, r2

import pandas as pd, numpy as np, subprocess, tempfile, pathlib, shutil

def adonis2_permanova_r2_via_rscript(X: np.ndarray, grouping, metric="braycurtis", permutations=999):
    X = pd.DataFrame(np.asarray(X, float))
    meta = pd.DataFrame({"group": np.asarray(grouping)})
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        Xf, Mf, Of, Rf = td/"X.csv", td/"meta.csv", td/"out.csv", td/"run.R"
        X.to_csv(Xf, index=False); meta.to_csv(Mf, index=False)
        Rf.write_text(f"""
        suppressMessages(library(vegan)); suppressMessages(library(compositions))
        X <- read.csv("{Xf}", check.names=FALSE); META <- read.csv("{Mf}")
        META$group <- factor(META$group)
        if ("{metric}"=="braycurtis") {{
          d <- vegdist(X, method="bray")
        }} else if ("{metric}"=="aitchison") {{
          X <- as.matrix(X) + 1e-6; X <- X/rowSums(X); X <- clr(X); d <- dist(X, method="euclidean")
        }} else stop("metric not supported")
        fit <- adonis2(d ~ group, data=META, permutations={permutations})
        write.csv(as.data.frame(fit), file="{Of}", row.names=TRUE)
        """, encoding="utf-8")
        subprocess.run(["Rscript", str(Rf)], check=True)
        res = pd.read_csv(Of, index_col=0)
        term_rows = [i for i in res.index if i not in ("Residual","Total")]
        r2 = float(res.loc[term_rows[0], "R2"])
        return res, r2
    
    