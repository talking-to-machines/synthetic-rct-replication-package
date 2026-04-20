# Figure and table map

Maps every figure and table in the paper to the script and input data
that generates it. All outputs are produced by `make analysis`.

---

## Figures

| Paper figure | Description | Script | Input data | Output file |
|---|---|---|---|---|
| Figure 1 | | `analysis/figures.R` | | `outputs/figures/fig_01_*.pdf` |
| Figure 2 | | `analysis/figures.R` | | `outputs/figures/fig_02_*.pdf` |
| Figure 3 | | `analysis/figures.R` | | `outputs/figures/fig_03_*.pdf` |
| Figure 4 | | `analysis/figures.R` | | `outputs/figures/fig_04_*.pdf` |
| Figure 5 | | `analysis/figures.R` | | `outputs/figures/fig_05_*.pdf` |

---

## Tables

| Paper table | Description | Script | Input data | Output file |
|---|---|---|---|---|
| Table 1 | | `analysis/tables.R` | | `outputs/tables/tab_01_*.tex` |
| Table 2 | | `analysis/tables.R` | | `outputs/tables/tab_02_*.tex` |
| Table 3 | | `analysis/tables.R` | | `outputs/tables/tab_03_*.tex` |
| Table 4 | | `analysis/tables.R` | | `outputs/tables/tab_04_*.tex` |
| Table 5 | | `analysis/tables.R` | | `outputs/tables/tab_05_*.tex` |

---

## Appendix figures and tables

| Item | Description | Script | Input data | Output file |
|---|---|---|---|---|
| Figure S1 | | `analysis/figures.R` | | `outputs/figures/fig_s01_*.pdf` |
| Table S1 | | `analysis/tables.R` | | `outputs/tables/tab_s01_*.tex` |

---

## Reproduction

```
make analysis
```

Generates all figures in `outputs/figures/` and all tables in
`outputs/tables/`. Individual scripts can also be run directly:

```
Rscript analysis/figures.R
Rscript analysis/tables.R
```

