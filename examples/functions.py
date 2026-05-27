from sklearn.metrics import silhouette_score
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import chi2

def extract_genus_from_taxonomy(taxonomy_series: pd.Series) -> pd.Series:
    genus = (
        taxonomy_series
        .astype(str)
        .str.extract(r"g__([^;]*)", expand=False)
        .fillna("")
        .str.strip()
    )

    genus_lower = genus.str.lower()
    genus = genus.mask(
        genus_lower.isin([
            "", "uncultured", "unassigned", "ambiguous_taxa", "incertae_sedis"
        ]),
        "Unassigned"
    )

    genus = genus.replace({
        "uncultured": "Unassigned",
        "unassigned": "Unassigned",
        "Ambiguous_taxa": "Unassigned",
        "Incertae_Sedis": "Unassigned",
    })

    return genus

def read_hivrc_raw(file_path: str):
    sample_id_col = "SeqID"

    otu_raw_path = Path(file_path) / "insight.merged_otus.txt"
    meta_raw_path = Path(file_path) / "SupplementaryMaterial.xlsx"

    otu_raw = pd.read_csv(
        otu_raw_path,
        sep="\t",
        encoding="utf-8",
        index_col="Resphera Insight (Raw Counts)"
    )
    meta_raw = pd.read_excel(meta_raw_path, header=1, usecols="B:L")

    otu_raw.index.name = sample_id_col
    otu_raw = otu_raw.reset_index()
    otu_raw.index = otu_raw[sample_id_col].astype(str)

    return otu_raw, meta_raw

def build_dataset(
    otu_raw: pd.DataFrame,
    meta_raw: pd.DataFrame,
    aggregation=None,
    subset_studies=None,
    
    cleanset_filtering = False,
    normalize = None,
    otu_zeroprev= None,
    sample_zeroprev=True,
    
    sample_id_col="SeqID",
    study_col="Study",
    hiv_col="hivstatus",
    age_col="Age",
    gender_col="gender",
    taxonomy_col="taxonomy",
):
    sample_cols = [c for c in otu_raw.columns if c not in [taxonomy_col, sample_id_col]]
    otu_counts = otu_raw[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    if aggregation == "genus":
        genus = extract_genus_from_taxonomy(otu_raw[taxonomy_col])
        otu_counts["Feature"] = genus.values
        feature_by_sample = otu_counts.groupby("Feature", dropna=False).sum()
    elif aggregation is None:
        otu_counts.index = otu_raw[sample_id_col].astype(str)
        feature_by_sample = otu_counts.copy()
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    # common sample filtering -----------------------------------------------------
    X = feature_by_sample.T
    X.index = X.index.astype(str)

    meta = meta_raw.copy()
    meta[sample_id_col] = meta[sample_id_col].astype(str)
    meta = meta.set_index(sample_id_col, drop=False)

    common_samples = X.index.intersection(meta.index)
    X = X.loc[common_samples].copy()
    meta = meta.loc[common_samples].copy()
    # -----------------------------------------------------------------------------

    # subset_studies --------------------------------------------------------------
    if subset_studies is not None:
        keep_subset = meta[study_col].isin(subset_studies)
        X = X.loc[keep_subset.values].copy()
        meta = meta.loc[keep_subset].copy()    
        batch_label = meta[study_col]
    # -----------------------------------------------------------------------------

    # cleanset_ filtering ----------------------------------------------------------
    if cleanset_filtering:
        meta[age_col] = pd.to_numeric(meta[age_col], errors="coerce")
        meta[study_col] = pd.Categorical(meta[study_col])

        keep_complete = meta[[study_col, hiv_col, age_col, gender_col]].notna().all(axis=1)
        X = X.loc[keep_complete.values].copy()
        meta = meta.loc[keep_complete].copy()

        X = X.loc[:, X.sum(axis=0) > 0].copy()

        meta = meta.reset_index(drop=True)
        X.index = meta[sample_id_col].values    
        batch_label = meta[study_col]
    # ------------------------------------------------------------------------------
    
    # zero_prevalence filtering -----------------------------------------------------
    if otu_zeroprev is not None:
        # prevalence filtering
        prevalence = (X > 0).sum(axis=0) / X.shape[0]
        X = X.loc[:, prevalence > otu_zeroprev].copy()

        # zero-library sample 제거
        row_sums = X.sum(axis=1)
        keep = row_sums > 0
        X = X.loc[keep].copy()
        meta = meta.loc[keep.values].reset_index(drop=True)
        batch_label = meta[study_col]
        batch_label = batch_label.loc[keep.values].reset_index(drop=True)
        
    if sample_zeroprev is not None:
        prevalence = (X > 0).sum(axis=1) / X.shape[1]
        keep_sample = prevalence > sample_zeroprev
        X = X.loc[keep_sample, :].copy()
        meta = meta.loc[keep_sample.values].reset_index(drop=True)
        batch_label = meta[study_col]
        batch_label = batch_label.loc[keep_sample.values].reset_index(drop=True)
                     
    # ------------------------------------------------------------------------------
    
            
    # normalization ----------------------------------------------------------------
    row_sums = X.sum(axis=1)
    keep_nonzero = row_sums > 0
    X = X.loc[keep_nonzero].copy()
    meta = meta.loc[keep_nonzero.values].reset_index(drop=True)
    batch_label = meta[study_col]
    batch_label = batch_label.loc[keep_nonzero.values].reset_index(drop=True)

    if normalize:
        X = X.div(X.sum(axis=1), axis=0)
    # ------------------------------------------------------------------------------

    
    # 정렬 보정
    X.index = meta["SeqID"].astype(str).values
    batch_label = meta[study_col]
                       
    return X, meta, batch_label

def get_experiment_data(
    design_id: str,
    file_path: str,
    verbose: bool = True,
):
    """
    design_id(df1 ~ df10)에 따라 feature_table, meta_data, batch_label 반환
    """

    if design_id not in EXPERIMENT_DESIGNS:
        raise ValueError(
            f"Unknown design_id: {design_id}. "
            f"Available: {list(EXPERIMENT_DESIGNS.keys())}"
        )

    cfg = EXPERIMENT_DESIGNS[design_id]

    normalize = cfg["normalize"]
    aggregation = cfg["aggregation"]
    cleanset_filtering = cfg["cleanset_filtering"]
    otu_zeroprev = cfg["otu_zeroprev"]
    sample_zeroprev = cfg["sample_zeroprev"]
    subset_studies = cfg["subset_studies"]
    

    otu_raw, meta_raw = read_hivrc_raw(file_path)

    X, meta_data, batch_label = build_dataset(
        otu_raw=otu_raw,
        meta_raw=meta_raw,
        aggregation=aggregation,
        subset_studies=subset_studies,
        normalize=normalize,
        cleanset_filtering=cleanset_filtering,
        otu_zeroprev=otu_zeroprev,
        sample_zeroprev=sample_zeroprev,
    )
    


    assert len(X) == len(meta_data) == len(batch_label)
    assert all(X.index == meta_data["SeqID"].astype(str))

    if verbose:
        print("=" * 70)
        print(f"Design ID               : {design_id}")
        print(f"Design name             : {cfg['name']}")
        print(f"Description             : {cfg['description']}")
        print(f"Aggregation             : {aggregation}")
        print(f"Normalize               : {normalize}")
        print(f"Cleanset Filtering      : {cleanset_filtering}")        
        print(f"Subset studies          : {subset_studies}")
        print(f"OTU zero-prev           : {otu_zeroprev}")
        print(f"Sample zero-prev        : {sample_zeroprev}")
        print("-" * 70)
        print(f"feature_table           : {X.shape}")
        print(f"meta_data               : {meta_data.shape}")
        print(f"n_batches               : {pd.Series(batch_label).nunique()}")
        print("=" * 70)

    return X, meta_data, batch_label, cfg

def compute_asw(X, labels, metric='euclidean'):
    """
    ASW (Average Silhouette Width)
    배치 기준: 낮을수록 (0에 가까울수록) 좋음
    생물학적 기준: 높을수록 좋음
    범위: -1 ~ 1
    """
    labels = np.array(labels).astype(str)
    # 레이블이 2개 이상일 때만 계산 가능
    if len(np.unique(labels)) < 2:
        return np.nan
    return silhouette_score(X, labels, metric=metric)

def evaluate_batch_correction(
    X_before,
    X_after,
    batch_labels,
    bio_labels=None,
    kbet_k=30,
    permanova_permutations=99,
    metric='euclidean',
    renormalize=False,   # ← 추가: relative abundance 데이터일 때 True
    label="Dataset"
):
    """
    배치 효과 보정 전후 성능 평가

    Parameters
    ----------
    X_before      : np.ndarray or pd.DataFrame — 보정 전 데이터 (samples × features)
    X_after       : np.ndarray or pd.DataFrame — 보정 후 데이터 (samples × features)
    batch_labels  : array-like — 배치 레이블 (Study 등)
    bio_labels    : array-like or None — 생물학적 레이블 (hivstatus 등)
    kbet_k        : int — kBET k-nearest neighbor 수
    permanova_permutations : int — PERMANOVA permutation 수
    metric        : str — 거리 metric
    label         : str — 출력 레이블

    Returns
    -------
    pd.DataFrame — 보정 전후 metric 결과표
    """

    if isinstance(X_before, pd.DataFrame):
        X_before = X_before.values
    if isinstance(X_after, pd.DataFrame):
        X_after = X_after.values
    
    # ← 추가: inf/NaN 처리 및 re-normalization
    def clean_and_normalize(X):
        X = pd.DataFrame(X)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
        row_sums = X.sum(axis=1).replace(0, 1)
        X = X.div(row_sums, axis=0)
        return X.values
    
    X_after = clean_and_normalize(X_after)

    batch_labels = np.array(batch_labels).astype(str)
    
    # ── NaN 마스크 처리
    if bio_labels is not None:
        bio_labels = np.array(bio_labels)
        bio_mask = ~pd.isna(bio_labels)
        bio_labels_clean = bio_labels[bio_mask].astype(str)
        X_before_bio = X_before[bio_mask]
        X_after_bio  = X_after[bio_mask]
    else:
        bio_mask = None

    results = {}

    # ════════════════════════════════
    # 1. PERMANOVA R² (batch)
    # ════════════════════════════════
    def permanova_r2(X, labels):
        dist = squareform(pdist(X, metric=metric))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        dm = DistanceMatrix(dist)
        
        # ← labels를 numpy array로 변환 (index 문제 해결)
        labels = np.array(labels).flatten().astype(str)
        
        permanova(dm, labels, permutations=permanova_permutations)
        n = len(labels)
        d2 = dist ** 2
        sst = d2[np.triu_indices(n, 1)].sum() / n
        ssw = 0.0
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            if len(idx) < 2:
                continue
            d2_g = d2[np.ix_(idx, idx)]
            ssw += d2_g[np.triu_indices(len(idx), 1)].sum() / len(idx)
        return float((sst - ssw) / sst)

    results["PERMANOVA R² (batch) ↓"] = {
        "Before": permanova_r2(X_before, batch_labels),
        "After":  permanova_r2(X_after,  batch_labels),
    }

    # PERMANOVA R² (bio)
    if bio_labels is not None:
        results["PERMANOVA R² (bio) ↑"] = {
            "Before": permanova_r2(X_before_bio, bio_labels_clean),
            "After":  permanova_r2(X_after_bio,  bio_labels_clean),
        }

    # ════════════════════════════════
    # 2. kBET acceptance rate (batch)
    # ════════════════════════════════
    def kbet(X, labels, k=kbet_k):
        n = len(X)
        unique, counts = np.unique(labels, return_counts=True)
        batch_freq = counts / n
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        _, indices = nbrs.kneighbors(X)
        accept = []
        for i in range(n):
            neighbor_labels = labels[indices[i]]
            observed = np.array([np.sum(neighbor_labels == b) for b in unique], dtype=float)
            expected = batch_freq * k
            chi2_stat = np.sum((observed - expected) ** 2 / (expected + 1e-8))
            p_val = 1 - chi2.cdf(chi2_stat, df=len(unique) - 1)
            accept.append(p_val > 0.05)
        return float(np.mean(accept))

    results["kBET acceptance rate ↑"] = {
        "Before": kbet(X_before, batch_labels),
        "After":  kbet(X_after,  batch_labels),
    }

    # ════════════════════════════════
    # 3. ASW (batch & bio)
    # ════════════════════════════════
    def asw(X, labels):
        labels = np.array(labels).astype(str)
        if len(np.unique(labels)) < 2:
            return np.nan
        return float(silhouette_score(X, labels, metric=metric))

    results["ASW (batch) → 0"] = {
        "Before": asw(X_before, batch_labels),
        "After":  asw(X_after,  batch_labels),
    }

    if bio_labels is not None:
        results["ASW (bio) ↑"] = {
            "Before": asw(X_before_bio, bio_labels_clean),
            "After":  asw(X_after_bio,  bio_labels_clean),
        }

    # ════════════════════════════════
    # 결과 DataFrame 정리
    # ════════════════════════════════
    df_result = pd.DataFrame(results).T
    df_result.index.name = "Metric"
    df_result["Change"] = df_result["After"] - df_result["Before"]
    df_result = df_result.round(4)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(df_result.to_string())
    print(f"{'='*55}\n")

    return df_result

def zero_filter(df, meta, batch, cutoff):
    prevalence = (df > 0).sum(axis=0) / df.shape[0]
    df = df.loc[:, prevalence > best_cutoff].copy()

    row_sums = df.sum(axis=1)
    keep_sample = row_sums > 0
    df = df.loc[keep_sample].copy()
    meta = meta.loc[keep_sample.values].reset_index(drop=True)
    batch = batch.loc[keep_sample.values].reset_index(drop=True)
        
    assert len(df) == len(meta) == len(batch)
    assert all(df.index.astype(str) == meta["SeqID"].astype(str))
    assert all(df.index.astype(str) == batch["SeqID"].astype(str))
    
    return df, meta, batch