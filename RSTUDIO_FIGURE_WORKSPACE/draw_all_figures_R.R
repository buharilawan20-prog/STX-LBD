# ============================================================
# STX-LBD: Draw all main figures in R
# ============================================================

library(tidyverse)
library(igraph)
library(ggraph)
library(tidygraph)
library(ggrepel)
library(scales)
library(patchwork)

BASE <- "/home/bhlabos/LBD/New/STX_CORPUS_ENRICHMENT/RSTUDIO_FIGURE_WORKSPACE"
FIG <- file.path(BASE, "R_figures")
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

theme_pub <- function() {
  theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 18, hjust = 0.5),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "black"),
      legend.title = element_text(face = "bold"),
      legend.position = "right"
    )
}

# ============================================================
# FIGURE 1: Dinoflagellate corpus year distribution
# ============================================================

dino_pre <- read_csv(file.path(BASE, "corpus/dino_pre2016.csv"), show_col_types = FALSE)
dino_post <- read_csv(file.path(BASE, "corpus/dino_post2015.csv"), show_col_types = FALSE)

dino_all <- bind_rows(
  dino_pre %>% mutate(period = "Pre-2016"),
  dino_post %>% mutate(period = "Post-2015")
) %>%
  mutate(year = as.numeric(year)) %>%
  filter(!is.na(year))

year_df <- dino_all %>%
  count(year)

p1 <- ggplot(year_df, aes(x = year, y = n)) +
  geom_col(fill = "#2C7FB8", width = 0.8) +
  geom_vline(xintercept = 2015, linetype = "dashed", linewidth = 1.1, color = "#D95F02") +
  labs(
    title = "Dinoflagellate STX corpus distribution with 2015 cutoff",
    x = "Year",
    y = "Number of papers"
  ) +
  scale_x_continuous(breaks = pretty_breaks(n = 20)) +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

ggsave(file.path(FIG, "Figure1_dino_year_distribution.png"), p1, width = 12, height = 7, dpi = 400)
ggsave(file.path(FIG, "Figure1_dino_year_distribution.pdf"), p1, width = 12, height = 7)

# ============================================================
# FIGURE 2: KG networks: dino pre, dino post, cyano
# ============================================================

plot_kg <- function(edge_file, title, outname, top_n = 120) {
  edges <- read_csv(edge_file, show_col_types = FALSE) %>%
    mutate(weight = as.numeric(weight)) %>%
    arrange(desc(weight)) %>%
    slice_head(n = top_n) %>%
    filter(!is.na(source), !is.na(target), source != target)

  nodes <- bind_rows(
    edges %>% select(name = source, type = source_type),
    edges %>% select(name = target, type = target_type)
  ) %>%
    distinct(name, .keep_all = TRUE)

  graph <- tbl_graph(nodes = nodes, edges = edges %>% select(from = source, to = target, weight), directed = FALSE)

  p <- ggraph(graph, layout = "fr") +
    geom_edge_link(aes(width = weight), alpha = 0.22, color = "grey45") +
    geom_node_point(aes(color = type, size = centrality_degree()), alpha = 0.9) +
    geom_node_text(aes(label = ifelse(centrality_degree() > quantile(centrality_degree(), 0.75), name, "")),
                   repel = TRUE, size = 3) +
    scale_edge_width(range = c(0.2, 2.5), guide = "none") +
    scale_size(range = c(2.5, 8), guide = "none") +
    labs(title = title, color = "Entity type") +
    theme_void(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 17, hjust = 0.5),
      legend.position = "right"
    )

  ggsave(file.path(FIG, paste0(outname, ".png")), p, width = 11, height = 8, dpi = 400)
  ggsave(file.path(FIG, paste0(outname, ".pdf")), p, width = 11, height = 8)
}

plot_kg(file.path(BASE, "kg_edges/dino_pre2016_semantic_edges.csv"),
        "Dinoflagellate pre-2016 semantic KG",
        "Figure2A_dino_pre2016_KG")

plot_kg(file.path(BASE, "kg_edges/dino_post2015_semantic_edges.csv"),
        "Dinoflagellate post-2015 semantic KG",
        "Figure2B_dino_post2015_KG")

plot_kg(file.path(BASE, "kg_edges/cyano_all_semantic_edges.csv"),
        "Cyanobacterial STX semantic KG",
        "Figure2C_cyano_KG")

# ============================================================
# FIGURE 3: Strict temporal validation Precision@K
# ============================================================

strict <- read_csv(file.path(BASE, "ml/strict_temporal_validation_metrics.csv"), show_col_types = FALSE)

p3 <- ggplot(strict, aes(x = K, y = `Precision@K`)) +
  geom_line(linewidth = 1.2, color = "#2C7FB8") +
  geom_point(size = 3, color = "#2C7FB8") +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(
    title = "Strict temporal validation of Node2Vec-ranked hypotheses",
    x = "Top K",
    y = "Precision@K"
  ) +
  theme_pub()

ggsave(file.path(FIG, "Figure3_strict_temporal_precision.png"), p3, width = 8, height = 6, dpi = 400)
ggsave(file.path(FIG, "Figure3_strict_temporal_precision.pdf"), p3, width = 8, height = 6)

# ============================================================
# FIGURE 4: Node2Vec vs AI ranking
# ============================================================

compare <- read_csv(file.path(BASE, "ml/node2vec_vs_ai_comparison_metrics.csv"), show_col_types = FALSE)

p4 <- ggplot(compare, aes(x = K, y = `Precision@K`, color = Method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format(accuracy = 1)) +
  labs(
    title = "Node2Vec versus supervised AI ranking",
    x = "Top K",
    y = "Precision@K",
    color = "Method"
  ) +
  theme_pub()

ggsave(file.path(FIG, "Figure4_node2vec_vs_ai_precision.png"), p4, width = 8, height = 6, dpi = 400)
ggsave(file.path(FIG, "Figure4_node2vec_vs_ai_precision.pdf"), p4, width = 8, height = 6)

# ============================================================
# FIGURE 5: Conserved vs divergent STX semantic biology
# ============================================================

cat_df <- read_csv(file.path(BASE, "cross_taxa/true_divergent_vs_conserved_category_counts.csv"), show_col_types = FALSE) %>%
  filter(Category != "Other") %>%
  select(Category, Conserved_Percent, Divergent_Percent, Conserved_Count, Divergent_Count) %>%
  pivot_longer(cols = c(Conserved_Percent, Divergent_Percent),
               names_to = "Relationship_class",
               values_to = "Percent") %>%
  mutate(
    Count = ifelse(Relationship_class == "Conserved_Percent", Conserved_Count, Divergent_Count),
    Relationship_class = recode(Relationship_class,
                                "Conserved_Percent" = "Conserved / transferred",
                                "Divergent_Percent" = "Divergent / cyano-only"),
    Category = factor(Category, levels = c("Environmental", "Evolutionary", "Gene-related", "Mechanistic"))
  )

p5 <- ggplot(cat_df, aes(x = Category, y = Percent, fill = Relationship_class)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, color = "black") +
  geom_text(aes(label = paste0(round(Percent, 1), "%\n(n=", Count, ")")),
            position = position_dodge(width = 0.8),
            vjust = -0.25,
            size = 4,
            fontface = "bold") +
  scale_y_continuous(limits = c(0, 105)) +
  labs(
    title = "Conserved versus divergent STX semantic biology",
    x = "Biological category",
    y = "Percentage of relationships",
    fill = "Relationship class"
  ) +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 25, hjust = 1, face = "bold"))

ggsave(file.path(FIG, "Figure5_true_conserved_vs_divergent.png"), p5, width = 11, height = 7, dpi = 400)
ggsave(file.path(FIG, "Figure5_true_conserved_vs_divergent.pdf"), p5, width = 11, height = 7)

# ============================================================
# FIGURE 6: Cross-taxa transfer categories
# ============================================================

transfer <- read_csv(file.path(BASE, "cross_taxa/cyano_plus_dino_pre2016_predicts_dino_post2015.csv"),
                     show_col_types = FALSE)

transfer_counts <- transfer %>%
  count(transfer_type) %>%
  arrange(desc(n))

p6 <- ggplot(transfer_counts, aes(x = reorder(transfer_type, n), y = n)) +
  geom_col(fill = "#2C7FB8", color = "black") +
  coord_flip() +
  labs(
    title = "Cross-taxa transfer categories",
    x = "Transfer category",
    y = "Number of post-2015 dinoflagellate edges"
  ) +
  theme_pub()

ggsave(file.path(FIG, "Figure6_cross_taxa_transfer_categories.png"), p6, width = 9, height = 6, dpi = 400)
ggsave(file.path(FIG, "Figure6_cross_taxa_transfer_categories.pdf"), p6, width = 9, height = 6)

# ============================================================
# FIGURE 7: Top validated/unvalidated hypothesis network
# ============================================================

ai <- read_csv(file.path(BASE, "ml/dino_pre2016_hypotheses_ai_ranked.csv"), show_col_types = FALSE)

top_valid <- ai %>%
  filter(Temporal_Label == 1) %>%
  arrange(desc(Final_AI_Rank_Score)) %>%
  slice_head(n = 10)

top_unvalid <- ai %>%
  filter(Temporal_Label == 0) %>%
  arrange(desc(Final_AI_Rank_Score)) %>%
  slice_head(n = 10)

hyp_net <- bind_rows(
  top_valid %>% mutate(Status = "Validated"),
  top_unvalid %>% mutate(Status = "Unvalidated")
)

edges_hyp <- hyp_net %>%
  transmute(from = Source, to = Target, Status, weight = Final_AI_Rank_Score)

nodes_hyp <- bind_rows(
  hyp_net %>% select(name = Source, type = Source_Type),
  hyp_net %>% select(name = Target, type = Target_Type)
) %>%
  distinct(name, .keep_all = TRUE)

graph_hyp <- tbl_graph(nodes = nodes_hyp, edges = edges_hyp, directed = FALSE)

p7 <- ggraph(graph_hyp, layout = "fr") +
  geom_edge_link(aes(linetype = Status, width = weight), color = "grey45", alpha = 0.75) +
  geom_node_point(aes(color = type, size = centrality_degree()), alpha = 0.95) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3.2, fontface = "bold") +
  scale_edge_width(range = c(0.8, 2.2), guide = "none") +
  scale_linetype_manual(values = c("Validated" = "solid", "Unvalidated" = "dashed")) +
  scale_size(range = c(4, 9), guide = "none") +
  labs(
    title = "Top dinoflagellate STX hypotheses: validated and unvalidated predictions",
    color = "Entity type",
    linetype = "Validation status"
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 17, hjust = 0.5),
    legend.position = "bottom"
  )

ggsave(file.path(FIG, "Figure7_top_hypothesis_network.png"), p7, width = 12, height = 8, dpi = 400)
ggsave(file.path(FIG, "Figure7_top_hypothesis_network.pdf"), p7, width = 12, height = 8)

# ============================================================
# FIGURE 8: Corpus composition bar plot
# ============================================================

corpus_summary <- tibble(
  Corpus = c("Dino pre-2016", "Dino post-2015", "Cyano all"),
  Records = c(nrow(dino_pre), nrow(dino_post), nrow(read_csv(file.path(BASE, "corpus/cyano_all.csv"), show_col_types = FALSE)))
)

p8 <- ggplot(corpus_summary, aes(x = Corpus, y = Records)) +
  geom_col(fill = "#2C7FB8", color = "black", width = 0.7) +
  geom_text(aes(label = Records), vjust = -0.4, fontface = "bold", size = 5) +
  labs(
    title = "STX corpus composition",
    x = NULL,
    y = "Number of records"
  ) +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1, face = "bold"))

ggsave(file.path(FIG, "Figure8_corpus_composition.png"), p8, width = 8, height = 6, dpi = 400)
ggsave(file.path(FIG, "Figure8_corpus_composition.pdf"), p8, width = 8, height = 6)

cat("\nAll R figures saved to:\n", FIG, "\n")
