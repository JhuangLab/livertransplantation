.libPaths(c("/cluster/home/jhuang/bin/R/Rlibrary/4.5.1_conda", 
            "/cluster/home/jhuang/.conda/envs/R-4.5.1/lib/R/library"))

suppressPackageStartupMessages({
  library(jhtools)
  library(openxlsx)
  library(reshape2)
  library(clusterProfiler)
  library(ggplot2)
  library(dplyr)
  library(limma)
  library(org.Hs.eg.db)
  library(tidyverse)
})

project <- "liver_multiomics"
species <- "human"
dataset <- "Figure"
workdir <- glue::glue("~/projects/{project}/analysis/{species}/{dataset}") %>% checkdir()
setwd(workdir)

dat_d <- "/cluster/home/jhuang/projects/liver/analysis/wangrongrong/human/protein/rds"
info_d <- "/cluster/home/jhuang/projects/liver/docs/wangrongrong/sampleinfo"
config_fn <- c("/cluster/home/xyzhang_jh/projects/liver/docs/wangrongrong/color.yaml")
fig_d <- glue::glue("{workdir}/Sfigure2") %>% checkdir()

pr_dat <- read_rds(glue::glue("{dat_d}/pr_exp_dat.rds"))
sample_info <- read_rds(glue::glue("{info_d}/sampleinfo.rds"))

pr_p2g <- as.data.frame(pr_dat)[,c("ID", "genename")]
pr_p2g <- drop_na(pr_p2g) %>% `row.names<-` (.$ID)

sample_info <- sample_info %>% as.data.frame() %>% dplyr::filter(regroup %notin% "unknow") %>%
  dplyr::select(c("sample_id", "regroup"))
pr_dat <- pr_dat %>% as.data.frame() %>% dplyr::select(c("genename", sample_info$sample_id)) %>%
  drop_na(genename)
pr_dat <- pr_dat[!duplicated(pr_dat$genename), ] %>% `row.names<-` (.$genename) %>% 
  dplyr::select(-"genename")

remove_sample <- c('con5','con15','rpre3','rpre4', 'rpre6', 'rpre7', 'rpo2', 'rpo8', 'rpo9', 'rpo13', 'rpo15')
pr_dat <- pr_dat[, colnames(pr_dat) %notin% remove_sample]
sample_info <- sample_info[sample_info$sample_id %notin% remove_sample, ]
sample_info <- sample_info[c(grep("con", sample_info$sample_id), 
                             grep("rpre", sample_info$sample_id), 
                             grep("rpo", sample_info$sample_id)),]
pr_dat <- pr_dat[, sample_info$sample_id]

# pre vs control
sample_info$regroup <- gsub("-", "_treat_", sample_info$regroup)
sample_pre_con <- sample_info %>% dplyr::filter(regroup %in% c("Control", "Rejection_treat_pre"))
pr_dat1 <- pr_dat[, sample_pre_con$sample_id]

#limma
design <- model.matrix(~0 + factor(sample_pre_con$regroup))
colnames(design) <- c("Control", "Rejection_treat_pre")
rownames(design) <- sample_pre_con$sample_id

contrast <- makeContrasts(Rejection_treat_pre - Control, levels = design)
fit <- lmFit(pr_dat1, design)
fit1 <- contrasts.fit(fit, contrast)
fit1 <- eBayes(fit1)

deg_pre_con <- topTable(fit1, number = Inf, sort.by = "logFC") %>%
  dplyr::filter(P.Value < 0.05) %>% 
  dplyr::filter(abs(logFC) > 1) %>%
  dplyr::arrange(desc(logFC))

gene_pre_con <- rownames(deg_pre_con)

up_gene1 <- rownames(deg_pre_con[which(deg_pre_con$logFC > 0),])
down_gene1 <- rownames(deg_pre_con[which(deg_pre_con$logFC < 0),])

reference_kegg <- read_csv("/cluster/home/danyang_jh/ref/kegg/human/kegg_hsa_all_pth_genes.csv")
reference_kegg <- reference_kegg[,c("pth_name","gene_name")]
# up_kegg
KEGG_res <- enricher(
  up_gene1,
  pAdjustMethod = "BH",
  TERM2GENE = reference_kegg,
)
dot1 <- enrichplot::dotplot(KEGG_res, showCategory = 10, font.size = 6)+
  theme(legend.key.size = unit(3, "mm"),
        axis.text.y = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.9),
        axis.title.x = element_text(size = 14)) + 
  labs(x= "Ratio", y = "", title = glue::glue("Rejection-treat-pre vs. Control up-regulated"))
ggsave(glue::glue("{fig_d}/KEGG_pre2con_up.svg"), dot1, width = 6.5, height = 6)

KEGG_res <- enricher(
  down_gene1,
  pAdjustMethod = "BH",
  TERM2GENE = reference_kegg,
)
dot1 <- enrichplot::dotplot(KEGG_res, showCategory = 10, font.size = 6)+
  theme(legend.key.size = unit(3, "mm"),
        axis.text.y = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.9),
        axis.title.x = element_text(size = 14)) + 
  labs(x= "Ratio", y = "", title = glue::glue("Rejection-treat-pre vs. Control down-regulated"))
ggsave(glue::glue("{fig_d}/KEGG_pre2con_down.svg"), dot1, width = 6.5, height = 6)

#===============================================================================
#===============================================================================
#===============================================================================
# pre vs post
sample_pre_post <- sample_info %>% dplyr::filter(regroup %in% c("Rejection_treat_pre", "Rejection_treat_post"))
pr_dat2 <- pr_dat[, sample_pre_post$sample_id]

#limma
design <- model.matrix(~0 + factor(sample_pre_post$regroup))
colnames(design) <- c("Rejection_treat_post", "Rejection_treat_pre")
rownames(design) <- sample_pre_post$sample_id

contrast <- makeContrasts(Rejection_treat_pre - Rejection_treat_post, levels = design)
fit <- lmFit(pr_dat2, design)
fit2 <- contrasts.fit(fit, contrast)
fit2 <- eBayes(fit2)

deg_pre_post <- topTable(fit2, number = Inf, sort.by = "logFC") %>%
  dplyr::filter(P.Value < 0.05) %>% 
  dplyr::filter(abs(logFC) > 1) %>%
  dplyr::arrange(desc(logFC))

gene_pre_post <- rownames(deg_pre_post)
up_gene2 <- rownames(deg_pre_post[which(deg_pre_post$logFC > 0),])
down_gene2 <- rownames(deg_pre_post[which(deg_pre_post$logFC < 0),])

KEGG_res <- enricher(
  up_gene2,
  pAdjustMethod = "BH",
  TERM2GENE = reference_kegg,
)
dot1 <- enrichplot::dotplot(KEGG_res, showCategory = 10, font.size = 6)+
  theme(legend.key.size = unit(3, "mm"),
        axis.text.y = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.9),
        axis.title.x = element_text(size = 14)) + 
  labs(x= "Ratio", y = "", title = glue::glue("Rejection-treat-pre vs. Rejection-treat-post up-regulated"))
ggsave(glue::glue("{fig_d}/KEGG_pre2post_up.svg"), dot1, width = 6.5, height = 6)

KEGG_res <- enricher(
  down_gene2,
  pAdjustMethod = "BH",
  TERM2GENE = reference_kegg,
)
dot1 <- enrichplot::dotplot(KEGG_res, showCategory = 10, font.size = 6)+
  theme(legend.key.size = unit(3, "mm"),
        axis.text.y = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        plot.title = element_text(size = 16, hjust = 0.85),
        axis.title.x = element_text(size = 14)) + 
  labs(x= "Ratio", y = "", title = glue::glue("Rejection-treat-pre vs. Rejection-treat-post down-regulated"))
ggsave(glue::glue("{fig_d}/KEGG_pre2post_down.svg"), dot1, width = 6.5, height = 6)