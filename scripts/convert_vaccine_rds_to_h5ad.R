#!/usr/bin/env Rscript
# Convert the SARS-CoV-2 vaccine CITE Seurat .rds (Zenodo 7555405) into flat,
# language-agnostic files that assemble_vaccine_h5ad.py turns into an AnnData.
# cascadir needs AnnData; the Zenodo object is R Seurat, so this is the bridge.
#
# Writes (into --out_dir):
#   counts.mtx        RNA counts (genes x cells, MatrixMarket sparse)
#   genes.csv         RNA feature names (row order of counts.mtx)
#   barcodes.csv      cell barcodes (col order of counts.mtx) == meta.csv row order
#   adt.mtx           ADT/surface-protein matrix (proteins x cells)  [if an ADT assay exists]
#   adt_names.csv     ADT feature names (row order of adt.mtx)
#   meta.csv          obj meta.data (one row per cell; first column = barcode)
#   convert_report.txt  what was detected (assays, dims)
#
# Requires SeuratObject (or Seurat) — the class definitions are needed to read a
# Seurat S4 object at all. GetAssayData() handles v4 (slot) and v5 (layer)
# transparently and restores dimnames. Aborts clearly if neither package loads
# (the cluster verification step checks this before the DAG runs).
#
# Usage:
#   Rscript scripts/convert_vaccine_rds_to_h5ad.R \
#       --rds /cs/.../SARSCoV2_Vaccine/raw/PBMC_vaccine_CITE.rds \
#       --out_dir /cs/.../SARSCoV2_Vaccine/raw/flat

suppressWarnings(suppressMessages(library(Matrix)))

# ---- args -----------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 1 && i < length(args)) return(args[i + 1])
  default
}
rds_path <- get_arg("--rds")
out_dir  <- get_arg("--out_dir")
if (is.null(rds_path) || is.null(out_dir)) {
  stop("usage: --rds <PBMC_vaccine_CITE.rds> --out_dir <dir>")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
report <- c()
log <- function(...) { msg <- paste0(...); cat(msg, "\n"); report <<- c(report, msg) }

# ---- load class defs (SeuratObject is the light one; Seurat depends on it) --
loaded <- FALSE
if (requireNamespace("SeuratObject", quietly = TRUE)) {
  suppressWarnings(suppressMessages(library(SeuratObject))); loaded <- TRUE
} else if (requireNamespace("Seurat", quietly = TRUE)) {
  suppressWarnings(suppressMessages(library(Seurat))); loaded <- TRUE
}
if (!loaded) {
  stop("Need SeuratObject or Seurat installed to read a Seurat .rds. ",
       "Install with: R -e 'install.packages(\"SeuratObject\")'")
}

log("[load] readRDS ", rds_path)
obj <- readRDS(rds_path)
log("[load] class: ", paste(class(obj), collapse = ","))

# ---- assay-data accessor (GetAssayData: layer for v5, slot for v4) ----------
get_mat <- function(obj, assay, which) {
  m <- tryCatch(SeuratObject::GetAssayData(obj, assay = assay, layer = which),
                error = function(e) NULL)
  if (is.null(m) || prod(dim(m)) == 0) {
    m <- tryCatch(SeuratObject::GetAssayData(obj, assay = assay, slot = which),
                  error = function(e) NULL)
  }
  m
}

all_assays <- tryCatch(SeuratObject::Assays(obj), error = function(e) names(obj@assays))
log("[assays] present: ", paste(all_assays, collapse = ", "))

pick <- function(cands, pool) {
  for (c in cands) {
    hit <- pool[tolower(pool) == tolower(c)]
    if (length(hit)) return(hit[1])
  }
  NULL
}
rna_assay <- pick(c("RNA", "SCT", "originalexp"), all_assays)
if (is.null(rna_assay)) rna_assay <- all_assays[1]
adt_assay <- pick(c("ADT", "Protein", "prot", "CITE", "AB", "TotalSeqA"), all_assays)
log("[assays] RNA -> ", rna_assay, " | ADT -> ", ifelse(is.null(adt_assay), "<none>", adt_assay))

# ---- RNA counts -----------------------------------------------------------
rna <- get_mat(obj, rna_assay, "counts")
if (is.null(rna)) rna <- get_mat(obj, rna_assay, "data")  # last resort: normalized
if (is.null(rna)) stop("Could not extract an RNA matrix from assay ", rna_assay)
rna <- as(rna, "CsparseMatrix")
log("[RNA] dim genes x cells = ", nrow(rna), " x ", ncol(rna),
    " (min=", round(min(rna), 3), ")")

genes <- rownames(rna)
barcodes <- colnames(rna)
if (is.null(genes))    genes    <- paste0("gene", seq_len(nrow(rna)))
if (is.null(barcodes)) barcodes <- paste0("cell", seq_len(ncol(rna)))

Matrix::writeMM(rna, file.path(out_dir, "counts.mtx"))
writeLines(genes,    file.path(out_dir, "genes.csv"))
writeLines(barcodes, file.path(out_dir, "barcodes.csv"))
log("[write] counts.mtx, genes.csv, barcodes.csv")

# ---- ADT (surface protein) ------------------------------------------------
if (!is.null(adt_assay)) {
  adt <- get_mat(obj, adt_assay, "data")          # prefer normalized (CLR) for gating
  if (is.null(adt)) adt <- get_mat(obj, adt_assay, "counts")
  if (!is.null(adt)) {
    adt <- as(as.matrix(adt), "CsparseMatrix")
    if (!is.null(colnames(adt)) && all(barcodes %in% colnames(adt))) {
      adt <- adt[, barcodes, drop = FALSE]          # align to RNA barcode order
    }
    adt_names <- rownames(adt)
    if (is.null(adt_names)) adt_names <- paste0("prot", seq_len(nrow(adt)))
    Matrix::writeMM(adt, file.path(out_dir, "adt.mtx"))
    writeLines(adt_names, file.path(out_dir, "adt_names.csv"))
    log("[ADT] dim prot x cells = ", nrow(adt), " x ", ncol(adt),
        " -> adt.mtx, adt_names.csv")
  } else {
    log("[ADT] assay present but no extractable matrix; skipping protein export")
  }
} else {
  log("[ADT] no ADT/protein assay detected; state labeling will fall back to RNA markers")
}

# ---- meta.data ------------------------------------------------------------
meta <- tryCatch(obj[[]], error = function(e) obj@meta.data)  # SeuratObject: obj[[]] = meta.data
meta <- cbind(barcode = rownames(meta), meta)
if (all(barcodes %in% meta$barcode)) {
  meta <- meta[match(barcodes, meta$barcode), , drop = FALSE]  # align to RNA barcode order
}
write.csv(meta, file.path(out_dir, "meta.csv"), row.names = FALSE)
log("[meta] ", nrow(meta), " cells x ", ncol(meta), " columns -> meta.csv")
log("[meta] columns: ", paste(colnames(meta), collapse = ", "))

writeLines(report, file.path(out_dir, "convert_report.txt"))
log("[done] flat files in ", out_dir)
