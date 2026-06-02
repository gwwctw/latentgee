
# conda activate r_baseline
# /home/slyang/miniconda3/envs/r_baseline/bin/R
library(ConQuR)
library(foreach)

# df4 데이터 로드
X_df4 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/X_df4_filtered_cutoff0.005.csv", row.names=1, check.names=FALSE)
meta_df4 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/meta_df4_filtered_cutoff0.005.csv", row.names=1)
batch_df4 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/batch_df4_filtered_cutoff0.005.csv", row.names=1)[,1]

batch_df4 <-relevel(as.factor(batch_df4), ref="Noguera-Julian")

# ConQuR 실행 (raw count 필요)
X_conqur_df4 <- ConQuR(
  tax_tab = X_df4,
  batchid = batch_df4,
  covariates = meta_df4[, c("hivstatus", "Age", "gender")],
  batch_ref = names(sort(table(batch_df4), decreasing=TRUE))[1]
)

# 저장
write.csv(X_conqur_df4, "/DATA/WGS_study/YSL/projects/latentgee/examples/data/X_df4_corrected_ConQuR.csv")
cat("ConQuR 완료\n")


# df6 데이터 로드
X_df6 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/X_df6_filtered_cutoff0.1.csv", row.names=1, check.names=FALSE)
meta_df6 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/meta_df6_filtered_cutoff0.1.csv", row.names=1)
batch_df6 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/batch_df6_filtered_cutoff0.1.csv", row.names=1)[,1]

batch_df6 <-relevel(as.factor(batch_df6), ref="Nowak2017")

# ConQuR 실행 (raw count 필요)
X_conqur_df6 <- ConQuR(
  tax_tab = X_df6,
  batchid = batch_df6,
  covariates = meta_df6[, c("hivstatus", "Age", "gender")],
  batch_ref = names(sort(table(batch_df6), decreasing=TRUE))[1]
)

# 저장
write.csv(X_conqur_df6, "/DATA/WGS_study/YSL/projects/latentgee/examples/results/baseline_results/X_df6_corrected_ConQuR.csv")
cat("ConQuR 완료\n")

