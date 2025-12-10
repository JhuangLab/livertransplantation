.libPaths(c("/cluster/home/jhuang/bin/R/Rlibrary/4.5.1_conda", 
            "/cluster/home/jhuang/.conda/envs/R-4.5.1/lib/R/library"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggsignif)
  library(jhtools)
  library(ggthemes)
  library(ggsci)
  library(reshape)
  library(openxlsx)
})

project <- "liver_multiomics"
species <- "human"
dataset <- "Figure1"

workdir <- glue::glue("~/projects/{project}/analysis/{species}/Figure/{dataset}/") %>% checkdir()
setwd(workdir)

remove_sample <- c('con5','con15','rpre3','rpre4', 'rpre6', 'rpre7', 'rpo2', 'rpo8', 'rpo9', 'rpo13', 'rpo15')
info_d <- "/cluster/home/jhuang/projects/liver/docs/wangrongrong/sampleinfo"

sample_info <- read_rds(glue::glue("{info_d}/sampleinfo.rds"))
sample_info <- sample_info[sample_info$sample_id %notin% remove_sample, ]
sample_info <- sample_info[c(grep("con", sample_info$sample_id), 
                             grep("rpre", sample_info$sample_id), 
                             grep("rpo", sample_info$sample_id)),]
sample_info$regroup <- gsub("-", "_treat_", sample_info$regroup)

sample_info <- as.data.frame(sample_info) %>% dplyr::filter(regroup %notin% c("unknow"))

#===============================================================================
#===============================================================================
#===============================================================================
# Sample gender info barplot 
sample_info$regroup <- gsub("_", "-", sample_info$regroup)
sample_info$regroup <- factor(sample_info$regroup, levels = c("Control", "Rejection-treat-pre", "Rejection-treat-post"))

p1 <- ggplot(sample_info, aes(x = regroup, fill = gender)) + 
  geom_bar(position = "dodge", width = 0.8, color = "black") + 
  theme_classic() +
  labs(fill = "Gender") +
  xlab(NULL) +
  ylab("Counts") +
  theme(axis.text.x = element_text(size = 16, color = "black", angle = 15, vjust = 0.5),
        axis.text.y = element_text(size = 16, color = "black"),
        axis.title.y = element_text(size = 18, color = "black"),
        legend.title = element_text(size = 16, color = "black"),
        legend.text = element_text(size = 14, color = "black")) + 
  scale_fill_manual(values = c("#90BFF9", "#F59092"))

pdf("./sample_gender_bar.pdf", height = 6, width = 6.5)
p1
dev.off()

svg("./sample_gender_bar.svg", height = 6, width = 6.5)
p1
dev.off()

#================================================================================
#================================================================================
#================================================================================
# Sample age info barplot
sample_info$age <- as.numeric(sample_info$age)

set.seed(2024)
age_p <- ggplot(sample_info, aes(x = regroup, y = age)) + 
  stat_boxplot(geom = "errorbar", width = 0.3, size = 0.8) + 
  geom_boxplot(aes(fill = regroup), outlier.colour = "white", linewidth = 0.8, width = 0.7) +
  geom_jitter(width = 0.3) + 
  theme_classic() +
  labs(fill = "Group") +
  xlab(NULL) +
  ylab("Age") +
  scale_y_continuous(limits = c(10,80)) + 
  theme(axis.text.x = element_text(size = 16, color = "black", angle = 15, vjust = 0.5),
        axis.text.y = element_text(size = 16, color = "black"),
        axis.title.y = element_text(size = 18, color = "black"),
        legend.title = element_text(size = 16, color = "black"),
        legend.text = element_text(size = 14, color = "black"),
        legend.position = "none") + 
  scale_fill_manual(values = c("#90BFF9", "#F2B77C", "#F59092"))

pdf("./sample_age_boxplot.pdf", height = 6, width = 6.5)
age_p
dev.off()

svg("./sample_age_boxplot.svg", height = 6, width = 6.5)
age_p
dev.off()