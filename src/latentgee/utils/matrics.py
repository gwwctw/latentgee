
# =========================
# Helper 함수
# =========================
def _safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str = "braycurtis") -> float:
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2 or np.any(counts < 2):
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric=metric))
    except Exception:
        return float("nan")

def _eval_block(X: np.ndarray, labels: np.ndarray, sil_metric: str, r2_metric: str,
                permutations: int, standardize: bool) -> Dict[str, float]:
    X_in = np.asarray(X, dtype=np.float64)
    if standardize:
        X_in = StandardScaler().fit_transform(X_in)
    sil = _safe_silhouette(X_in, labels, metric=sil_metric)
    _, r2 = permanova_r2(X_in, grouping=np.asarray(labels), metric=r2_metric, permutations=permutations)
    return {"silhouette": float(sil), "r2": float(r2)}