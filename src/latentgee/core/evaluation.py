import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova


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
    

def evaluate_latentgee(
    model: torch.nn.Module,
    X_tensor: torch.Tensor,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    allow_noise: bool = True,          # True면 노이즈 포함, False면 -1 제외
    metric: str = "braycurtis",
) -> Tuple[np.ndarray, float]:
    """
    Encoder의 μ → HDBSCAN으로 pseudo-batch → GEE residual → silhouette 반환.
    labels: np.ndarray (길이 N, -1 = noise)
    score : float (noise 포함/제외는 allow_noise에 따름; 실패 시 -1.0)
    """
    # 1) 인코딩
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        mu, _ = model.encode(X_tensor.to(device, non_blocking=(device.type == "cuda")))
    z_np = mu.detach().cpu().numpy().astype("float32")

    # (선택) 표준화: HDBSCAN/실루엣 안정화
    z_np = StandardScaler().fit_transform(z_np)

    # 2) HDBSCAN (문자열 "None" 방지)
    if isinstance(min_samples, str) and min_samples.lower() == "none":
        min_samples = None

    hdb_kwargs = dict(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method="eom",
    )
    try:
        # cosine은 BallTree 미지원 → generic로 시도
        if metric == "cosine":
            hdb = HDBSCAN(algorithm="generic", **hdb_kwargs)
        else:
            hdb = HDBSCAN(**hdb_kwargs)
        labels = hdb.fit_predict(z_np)
    except ValueError:
        warnings.warn(
            f"[WARN] HDBSCAN metric '{metric}' not supported → fallback to 'euclidean'",
            RuntimeWarning,
        )
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = hdb.fit_predict(z_np)

    # 3) 사용 샘플 선택 (allow_noise 의미 정정)
    if allow_noise:
        mask = np.ones_like(labels, dtype=bool)   # 노이즈 포함
    else:
        mask = (labels != -1)                     # 노이즈 제외

    n_valid_clusters = np.unique(labels[mask]).size
    n_valid_points   = int(mask.sum())

    # 4) HDBSCAN 결과가 부적절하면 KMeans 폴백(= best-k 탐색 대신 k=2라도 OK)
    #    → best-k를 쓰고 싶으면 별도 함수 호출로 교체 가능
    if n_valid_clusters < 2 or n_valid_points < 3:
        # KMeans로 강제 라벨링 (원하면 find_best_k_silhouette로 교체)
        if z_np.shape[0] >= 2:
            km = KMeans(n_clusters=2, n_init="auto", random_state=42).fit(z_np)
            labels = km.labels_
            # KMeans 라벨엔 노이즈 개념 없음 → 전부 사용
            z_used, lbl_used = z_np, labels
        else:
            return labels, -1.0
    else:
        # 마스크에 따라 사용 데이터 결정
        z_used, lbl_used = z_np[mask], labels[mask]

    # 5) GEE residual → silhouette
    z_resid = gee_latent_residual(z_used, lbl_used)  # -1 라벨은 위에서 제거됨
    try:
        sil = silhouette_score(z_resid, lbl_used, metric=metric)
    except Exception:
        # 거리 메트릭 이슈 → euclidean 폴백
        try:
            sil = silhouette_score(z_resid, lbl_used, metric="euclidean")
        except Exception:
            sil = -1.0

    return labels, float(sil)

# -----------------------
# Evaluator (silhouette / PERMANOVA R²)
# -----------------------
class BatchEffectEvaluator:
    def __init__(self, eval_cfg: EvalConfig):
        self.cfg = eval_cfg

    @torch.no_grad()
    def silhouette_from_model(self, model: LatentGEEModule, X: np.ndarray) -> Tuple[np.ndarray, float]:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        labels, sil = evaluate_latentgee(
            model=model.vae,
            X_tensor=X_tensor,
            min_cluster_size=self.cfg.hdb_min_cluster_size,
            min_samples=self.cfg.hdb_min_samples,
            allow_noise=self.cfg.allow_noise,
            metric=self.cfg.hdb_metric,
        )
        return labels, float(sil)

    def permanova_r2_from_matrix(self, X: np.ndarray, labels: np.ndarray) -> Tuple[Any, float]:
        X_std = StandardScaler().fit_transform(X)
        res, r2 = permanova_r2(
            X_std,
            grouping=labels,
            metric=self.cfg.permanova_metric,
            permutations=self.cfg.permanova_permutations,
        )
        return res, float(r2)