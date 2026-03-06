# Table 3: Inflation Schedule Calibration

**Decision:** Are the heuristic inflation factors reasonable?

Empirical factors derived by subsampling high-N pairs (n≥30) to target sizes and measuring CV_true / CV_subsample.

| Target n | Heuristic | Empirical median | p25 | p75 | p90 | Assessment | Pairs used |
|:--------:|:---------:|:----------------:|:---:|:---:|:---:|:-----------|:----------:|
| 2 | 2.5× | 2.74× | 1.36× | 6.25× | 17.11× | Slightly under-corrects | 11 |
| 3 | 2.0× | 1.76× | 1.04× | 3.30× | 6.29× | Reasonable (between median and p75) | 11 |
| 5 | 1.5× | 1.33× | 0.92× | 2.12× | 3.41× | Reasonable (between median and p75) | 11 |
| 10 | 1.2× | 1.10× | 0.88× | 1.44× | 2.18× | Reasonable (between median and p75) | 11 |

**Interpretation:** If heuristic > empirical median, the schedule is conservative (wider uncertainty than data suggests). If heuristic < empirical median, the schedule under-corrects (uncertainty is still too narrow after inflation).