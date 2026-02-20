
# prevalence cutoff에 따른 zero 비율 계산 ---
def compute_zero_proportion_by_prevalence(otu_df, cutoffs):
    n_samples = otu_df.shape[0]
    proportions = []

    for cutoff in cutoffs:
        # prevalence 계산: 각 OTU의 nonzero 비율
        prevalence = (otu_df > 0).sum(axis=0) / n_samples

        # cutoff 미만인 OTU 제거
        filtered_df = otu_df.loc[:, prevalence >= cutoff]

        # 전체 zero 비율 계산
        zero_count = (filtered_df == 0).sum().sum()
        total_count = filtered_df.size
        zero_proportion = zero_count / total_count

        proportions.append(zero_proportion)

    return proportions

# 베스트 모델 저장/재현
def save_model(model, path="best_model.pt"):
    torch.save(model.state_dict(), path)

def load_model(model_class, path="best_model.pt", *args, device=None,**kwargs):
    model=model_class(*args, **kwargs)
    ckpt=torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt)
    if device: model.to(device)
    model.eval()
    return model