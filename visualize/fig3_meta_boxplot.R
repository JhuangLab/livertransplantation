.libPaths(c("/cluster/home/jhuang/bin/R/Rlibrary/4.5.1_conda", 
            "/cluster/home/jhuang/.conda/envs/R-4.5.1/lib/R/library"))

suppressPackageStartupMessages({
  library(jhtools)
  library(openxlsx)
  library(reshape2)
  library(ggsignif)
  library(ggplot2)
  library(dplyr)
  library(tidyverse)
})
project <- "liver_multiomics"
species <- "human"
dataset <- "Figure"
workdir <- glue::glue("~/projects/{project}/analysis/{species}/{dataset}/Figure3") %>% checkdir()
setwd(workdir)

remove_sample <- c('con5','con15','rpre3','rpre4', 'rpre6', 'rpre7', 'rpo2', 'rpo8', 'rpo9', 'rpo13', 'rpo15')
meta_d <- "/cluster/home/jhuang/projects/liver/analysis/wangrongrong/human/metabolome/rds"
info_d <- "/cluster/home/jhuang/projects/liver/docs/wangrongrong/sampleinfo"
config_fn <- c("/cluster/home/xyzhang_jh/projects/liver/docs/wangrongrong/color.yaml")

sample_info <- read_rds(glue::glue("{info_d}/sampleinfo.rds"))
sample_info <- sample_info[sample_info$sample_id %notin% remove_sample, ]
sample_info <- sample_info[c(grep("con", sample_info$sample_id), 
                             grep("rpre", sample_info$sample_id), 
                             grep("rpo", sample_info$sample_id)),]
sample_info$regroup <- gsub("-", "_treat_", sample_info$regroup)
sample_pre_con <- sample_info %>% dplyr::filter(regroup %in% c("Control", "Rejection_treat_pre")) %>%
  dplyr::select(c("sample_id", "regroup"))

meta_dat <- read_rds(glue::glue("{meta_d}/mob_exp_dat.rds"))
meta_pre_con <- c(
  "neg-M267T318_1",
  "neg-M274T426_1",
  "neg-M274T426_2",
  "neg-M277T343",
  "neg-M336T427"
  )
meta_pre_con_mat <- meta_dat %>% dplyr::filter(ID %in% meta_pre_con) %>%
  dplyr::select(c("ID","MS2Metabolite", sample_pre_con$sample_id))
meta_pre_con_mat$MS2Metabolite[which(meta_pre_con_mat$ID %in% c("neg-M274T426_2"))] <- "Arg-Thr-2"

plot_df <- melt(meta_pre_con_mat, id = c("MS2Metabolite"))
plot_df <- merge(plot_df, sample_pre_con, by.x = "variable", by.y = "sample_id")
plot_df$regroup <- gsub("_", "-", plot_df$regroup)
plot_df$regroup <- factor(plot_df$regroup, levels = c("Control", "Rejection-treat-pre"))
plot_df$value <- as.double(plot_df$value) %>% round(digits = 2)

meta_list_pre_con <- c("Arg-Thr", "Arg-Thr-2","Phe-Leu", "CMPF", "Olopatadine")
plot_df$MS2Metabolite <- gsub("3-carboxy-4-methyl-5-pentyl-2-furanpropanoic acid", "CMPF", plot_df$MS2Metabolite)

p1 <- ggplot(plot_df, aes(x = regroup, y = value)) +
  stat_boxplot(geom = "errorbar", width = 0.2, size = 0.8) + 
  geom_boxplot(aes(fill = regroup), outlier.colour = "white", size = 0.8) +
  labs(title = NULL, fill = "Group") +
  ylab("Intensity") + 
  theme_bw() + 
  facet_wrap(.~MS2Metabolite, nrow = 1, scales = "free_y") +
  theme(panel.background = element_blank(),
        axis.line = element_line(),
        legend.position = "top",
        legend.title =  element_text(color = "black", size = 16), 
        legend.text = element_text(color = "black", size = 15), 
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.title.y = element_text(size = 20, color = "black"),
        strip.text = element_text(color = "black", size = 18),
        axis.ticks.x = element_blank()
        ) +
  geom_jitter(width = 0.2) +
  xlab(NULL) +
  geom_signif(comparisons = list(c("Rejection-treat-pre", "Control")),
              map_signif_level = T,
              size = 0.8,
              color = "black",
              textsize = 6, vjust = 0.5) + 
  scale_fill_manual(values = c("#90BFF9", "#F2B77C"))  
pdf("./boxplot/pre_con_meta_h5.pdf", height = 5, width = 14)
print(p1)
dev.off()

svg("./boxplot/pre_con_meta_h5_free_y.svg", height = 5, width = 14)
print(p1)
dev.off()
#===============================================================================
#===============================================================================
#===============================================================================
# pre vs post
sample_pre_post <- sample_info %>% dplyr::filter(regroup %in% c("Rejection_treat_pre", "Rejection_treat_post")) %>%
  dplyr::select(c("sample_id", "regroup"))
meta_dat <- read_rds(glue::glue("{meta_d}/mob_exp_dat.rds"))
meta_pre_post <- c(
  "neg-M329T470",
  "neg-M413T423",
  "pos-M307T370",
  "pos-M361T461",
  "pos-M363T469"
)
meta_pre_post_mat <- meta_dat %>% dplyr::filter(ID %in% meta_pre_post) %>%
  dplyr::select(c("MS2Metabolite", sample_pre_post$sample_id)) 

plot_df <- melt(meta_pre_post_mat, id = c("MS2Metabolite"))
plot_df <- merge(plot_df, sample_pre_post, by.x = "variable", by.y = "sample_id")
plot_df$regroup <- gsub("_", "-", plot_df$regroup)
plot_df$regroup <- factor(plot_df$regroup, levels = c("Rejection-treat-post", "Rejection-treat-pre"))
plot_df$value <- as.double(plot_df$value) %>% round(digits = 2)

meta_list_pre_con <- c("Fluconazole", "Cortisone", "Hydrocortisone", "Carnosol", "C21H34O6S")
plot_df$MS2Metabolite <- gsub("5.alpha.-Pregnan-3.alpha.,17-diol-20-one 3-sulfate", "C21H34O6S", plot_df$MS2Metabolite)

p1 <- ggplot(plot_df, aes(x = regroup, y = value)) +
  stat_boxplot(geom = "errorbar", width = 0.2, size = 0.8) + 
  geom_boxplot(aes(fill = regroup), outlier.colour = "white", size = 0.8) +
  labs(title = NULL, fill = "Group") +
  ylab("Intensity") + 
  theme_bw() + 
  facet_wrap(.~MS2Metabolite, scales = "free_y", nrow = 1)+
  theme(panel.background = element_blank(),
        axis.line = element_line(),
        legend.position = "top",
        legend.title =  element_text(color = "black", size = 16), 
        legend.text = element_text(color = "black", size = 15), 
        axis.text.x = element_blank(),
        axis.text.y = element_text(size = 14, color = "black"),
        axis.title.y = element_text(size = 20, color = "black"),
        strip.text = element_text(color = "black", size = 18),
        axis.ticks.x = element_blank()
  ) +
  geom_jitter(width = 0.2) +
  xlab(NULL) +
  geom_signif(comparisons = list(c("Rejection-treat-pre", "Rejection-treat-post")),
              map_signif_level = T,
              size = 0.8,
              color = "black",
              textsize = 6, vjust = 0.5) + 
  scale_fill_manual(values = c("#F59092", "#F2B77C"))  
pdf("./boxplot/pre_post_meta_h5_free_y.pdf", height = 5, width = 14)
print(p1)
dev.off()

svg("./boxplot/pre_post_meta_h5_free_y.svg", height = 5, width = 14)
print(p1)
dev.off()