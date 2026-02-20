
# ====== 2) 데이터 로더(Zero prevalence cutoff 캐시 지원) ======
# 전역 캐시 (메모리 or 디스크)
_DATASET_CACHE = {}

def get_dataset_for_cutoff(cutoff: float):
    """
    cutoff별 X_tensor를 CPU 텐서로 반환하고 input_dim을 리턴.
    1) 메모리 캐시 있으면 바로 사용
    2) 없으면 디스크 캐시(파일) 있나 확인 -> 로드
    3) 둘 다 없으면 원시데이터로부터 전처리 수행 -> 캐시/저장
    """
    key = f"zp_{cutoff:.4f}"
    if key in _DATASET_CACHE:
        X = _DATASET_CACHE[key]
        return X, X.shape[1]

    # (A) 디스크 캐시가 있다면:
    pkl_path = f".../preprocessed/hivrc_scene2_zp{cutoff:.2f}.pt"
    if os.path.exists(pkl_path):
        X = torch.load(pkl_path, map_location="cpu")
        _DATASET_CACHE[key] = X
        return X, X.shape[1]

    # (B) 원시데이터로부터 전처리 (여기서는 의사코드)
    # raw = load_raw(...)
    # X_np = preprocess_by_zero_prevalence(raw, cutoff=cutoff)  # np.ndarray (N,D)
    # X = torch.tensor(X_np, dtype=torch.float32)
    # torch.save(X, pkl_path)
    # _DATASET_CACHE[key] = X
    # return X, X.shape[1]
    raise FileNotFoundError(f"no dataset for cutoff={cutoff}; add preprocessing or cached file.")
