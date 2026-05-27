
# conda activate r_baseline
# /home/slyang  
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


# df4 데이터 로드
X_df5 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/X_df5_filtered_cutoff0.05.csv", row.names=1, check.names=FALSE)
meta_df5 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/meta_df5_filtered_cutoff0.05.csv", row.names=1)
batch_df5 <- read.csv("/DATA/WGS_study/YSL/projects/latentgee/examples/data/batch_df5_filtered_cutoff0.05.csv", row.names=1)[,1]

batch_df5 <-relevel(as.factor(batch_df5), ref="Nowak2017")

# ConQuR 실행 (raw count 필요)
X_conqur_df5 <- ConQuR(
  tax_tab = X_df5,
  batchid = batch_df5,
  covariates = meta_df5[, c("hivstatus", "Age", "gender")],
  batch_ref = names(sort(table(batch_df5), decreasing=TRUE))[1]
)

# 저장
write.csv(X_conqur_df5, "/DATA/WGS_study/YSL/projects/latentgee/examples/data/X_df5_corrected_ConQuR.csv")
cat("ConQuR 완료\n")

