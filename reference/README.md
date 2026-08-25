# Published reference data

The files in this directory are the compact reference data used for automatic
reproduction checks and for the tracked article outputs.

- `trials/main.csv` contains the 8,800 paired central-evaluation trials.
- `trials/lagrangian.csv` contains the 1,200 Lagrangian-baseline trials.
- `main_results.csv` contains the values displayed in Table 6 and Figure 4.
- `studies/` contains the values displayed in Figures 2, 3, and 5.
- `runtime/` contains the 2,880 native benchmark blocks and their published
  summary for Table 9.

The evaluation command compares newly generated central and Lagrangian trial
records with the two trial-level reference files. The study CSV files and
runtime measurements remain small enough to be stored directly in Git.
