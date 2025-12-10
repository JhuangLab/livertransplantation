#!/usr/bin/env Rscript

# If not installed, run once:
# BiocManager::install(c("clusterProfiler", "org.Hs.eg.db", "ReactomePA", "enrichplot"))

library(clusterProfiler)
library(org.Hs.eg.db)
library(ReactomePA)
library(enrichplot)
library(ggplot2)

# 1. Read feature table split by comparison
features_file <- "top50_features_two_comparisons/features_by_omics_and_comparison.csv"
features_df <- read.csv(features_file, stringsAsFactors = FALSE)

output_dir <- "protein_enrichment_results"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Extract gene symbol list from a specific column
extract_genes <- function(column_name) {
    if (!(column_name %in% colnames(features_df))) {
        return(character())
    }
    genes <- unique(features_df[[column_name]])
    genes <- genes[!is.na(genes) & genes != ""]
    return(genes)
}

# Columns for each comparison
group_columns <- list(
    Control_vs_Pre = "protein_control_pre_mapped",
    Pre_vs_Post    = "protein_pre_post_mapped"
)


# Main loop over comparisons
for (group_name in names(group_columns)) {
    genes <- extract_genes(group_columns[[group_name]])
    if (length(genes) == 0) {
        message(sprintf("%s: no genes available, skipped.", group_name))
        next
    }
    
    cat(sprintf("%s: %d genes.\n", group_name, length(genes)))
    
    gene_df <- bitr(
        genes,
        fromType = "SYMBOL",
        toType   = "ENTREZID",
        OrgDb    = org.Hs.eg.db
    )
    if (nrow(gene_df) == 0) {
        message(sprintf("%s: could not convert any Entrez IDs, skipped.", group_name))
        next
    }
    
    entrez_ids <- unique(gene_df$ENTREZID)
    cat(sprintf("%s: %d Entrez IDs after conversion.\n", group_name, length(entrez_ids)))
    
    group_dir <- file.path(output_dir, group_name)
    if (!dir.exists(group_dir)) dir.create(group_dir, recursive = TRUE)
    
    ekegg <- enrichKEGG(
        gene          = entrez_ids,
        organism      = "hsa",
        pvalueCutoff  = 0.05
    )
    result_df <- as.data.frame(ekegg)
    write.csv(result_df, file.path(group_dir, "KEGG_enrichment.csv"), row.names = FALSE)
    
    plot_path <- file.path(group_dir, "KEGG_bubble.png")
    plotted <- plot_kegg_dotplot(result_df, group_name, plot_path)
    if (!plotted) {
        message(sprintf("%s: empty KEGG result, plot not generated.", group_name))
    } else {
        message(sprintf("%s: plot saved to %s", group_name, plot_path))
    }
}