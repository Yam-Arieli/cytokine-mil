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
#   convert_report.txt  what was detected (assays, dims, access path)
#
# Two read paths:
#  (A) if SeuratObject/Seurat is installed -> GetAssayData() (handles v4 slot + v5 layer).
#  (B) otherwise -> BASE-R via attr(): an S4 object's slots survive readRDS as
#      attributes even without the class definition loaded, so we dig the assay
#      matrices out directly. Only the Matrix package is required (the inner count
#      matrices are dgCMatrix, whose class IS defined by Matrix). This avoids any
#      compile/install on the cluster (SeuratObject is not installed there).
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

have_so <- requireNamespace("SeuratObject", quietly = TRUE)
if (have_so) {
  suppressWarnings(suppressMessages(library(SeuratObject)))
} else if (requireNamespace("Seurat", quietly = TRUE)) {
  suppressWarnings(suppressMessages(library(Seurat))); have_so <- TRUE
}
log("[env] SeuratObject/Seurat available: ", have_so, " (FALSE -> base-R attr path)")

log("[load] readRDS ", rds_path)
obj <- readRDS(rds_path)
log("[load] class: ", paste(class(obj), collapse = ","))

# ---- accessors (path A: Seurat; path B: base-R attr) ----------------------
get_assay_names <- function(obj) {
  if (have_so) {
    nm <- tryCatch(SeuratObject::Assays(obj), error = function(e) NULL)
    if (!is.null(nm)) return(nm)
  }
  names(attr(obj, "assays"))
}

# Pull a matrix ('counts' or 'data') from one assay, by either path.
get_matrix <- function(obj, assay, which) {
  if (have_so) {
    m <- tryCatch(SeuratObject::GetAssayData(obj, assay = assay, layer = which),
                  error = function(e) NULL)
    if (is.null(m) || prod(dim(m)) == 0)
      m <- tryCatch(SeuratObject::GetAssayData(obj, assay = assay, slot = which),
                    error = function(e) NULL)
    if (!is.null(m) && prod(dim(m)) > 0) return(m)
  }
  # base-R attr path
  ao <- attr(obj, "assays")[[assay]]
  if (is.null(ao)) return(NULL)
  m <- attr(ao, which)                       # v4 Assay: @counts / @data
  if (is.null(m)) {
    ly <- attr(ao, "layers")                 # v5 Assay5: @layers list
    if (!is.null(ly)) m <- ly[[which]]
  }
  if (is.null(m)) return(NULL)
  # restore dimnames for v5 layers (stored separately in @features / @cells LogMaps)
  if (is.null(rownames(m))) {
    fe <- attr(ao, "features"); rn <- tryCatch(rownames(fe), error = function(e) NULL)
    if (!is.null(rn) && length(rn) == nrow(m)) rownames(m) <- rn
  }
  if (is.null(colnames(m))) {
    ce <- attr(ao, "cells"); cn <- tryCatch(rownames(ce), error = function(e) NULL)
    if (!is.null(cn) && length(cn) == ncol(m)) colnames(m) <- cn
  }
  m
}

get_meta <- function(obj) {
  if (have_so) {
    md <- tryCatch(obj[[]], error = function(e) NULL)
    if (!is.null(md)) return(md)
  }
  attr(obj, "meta.data")
}

# ---- pick assays ----------------------------------------------------------
all_assays <- get_assay_names(obj)
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
rna <- get_matrix(obj, rna_assay, "counts")
if (is.null(rna)) rna <- get_matrix(obj, rna_assay, "data")   # last resort: normalized
if (is.null(rna)) stop("Could not extract an RNA matrix from assay ", rna_assay)
rna <- as(rna, "CsparseMatrix")
log("[RNA] dim genes x cells = ", nrow(rna), " x ", ncol(rna), " (min=", round(min(rna), 3), ")")

meta <- get_meta(obj)
genes <- rownames(rna)
barcodes <- colnames(rna)
if (is.null(genes))    genes    <- paste0("gene", seq_len(nrow(rna)))
if (is.null(barcodes)) barcodes <- if (!is.null(meta)) rownames(meta) else paste0("cell", seq_len(ncol(rna)))

Matrix::writeMM(rna, file.path(out_dir, "counts.mtx"))
writeLines(genes,    file.path(out_dir, "genes.csv"))
writeLines(barcodes, file.path(out_dir, "barcodes.csv"))
log("[write] counts.mtx, genes.csv, barcodes.csv")

# ---- ADT (surface protein) ------------------------------------------------
if (!is.null(adt_assay)) {
  adt <- get_matrix(obj, adt_assay, "data")    # prefer normalized (CLR) for gating
  if (is.null(adt)) adt <- get_matrix(obj, adt_assay, "counts")
  if (!is.null(adt)) {
    adt <- as(as.matrix(adt), "CsparseMatrix")
    if (!is.null(colnames(adt)) && all(barcodes %in% colnames(adt)))
      adt <- adt[, barcodes, drop = FALSE]       # align to RNA barcode order
    adt_names <- rownames(adt)
    if (is.null(adt_names)) adt_names <- paste0("prot", seq_len(nrow(adt)))
    Matrix::writeMM(adt, file.path(out_dir, "adt.mtx"))
    writeLines(adt_names, file.path(out_dir, "adt_names.csv"))
    log("[ADT] dim prot x cells = ", nrow(adt), " x ", ncol(adt), " -> adt.mtx, adt_names.csv")
  } else {
    log("[ADT] assay present but no extractable matrix; skipping protein export")
  }
} else {
  log("[ADT] no ADT/protein assay detected; state labeling will fall back to RNA markers")
}

# ---- meta.data ------------------------------------------------------------
if (is.null(meta)) stop("Could not extract meta.data from the object.")
meta <- cbind(barcode = rownames(meta), meta)
if (all(barcodes %in% meta$barcode))
  meta <- meta[match(barcodes, meta$barcode), , drop = FALSE]  # align to RNA barcode order
write.csv(meta, file.path(out_dir, "meta.csv"), row.names = FALSE)
log("[meta] ", nrow(meta), " cells x ", ncol(meta), " columns -> meta.csv")
log("[meta] columns: ", paste(colnames(meta), collapse = ", "))

writeLines(report, file.path(out_dir, "convert_report.txt"))
log("[done] flat files in ", out_dir)
