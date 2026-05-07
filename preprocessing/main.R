# Preprocessing pipeline entry point.
#
# Sources shared helpers from utils.R, then runs each per-source cleaning
# script under rcts/ and surveys/. Per-source scripts read from
# data/human/{rcts,surveys}/{source_id}/ and write to
# data/processed/{rcts,surveys}/{source_id}/.
#
# To add a new source: drop a file named {source_id}_preprocess.R into the
# appropriate subfolder. It will be picked up automatically.

source("preprocessing/utils.R")

source_dir <- function(dir) {
  scripts <- list.files(dir, pattern = "\\.R$", full.names = TRUE)
  for (script in scripts) source(script)
}

source_dir("preprocessing/rcts")
source_dir("preprocessing/surveys")
