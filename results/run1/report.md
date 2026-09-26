## Per-arm results

| arm | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms | $/1k queries | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| embed_only | 175 | 0.532 | 0.842 | 0.913 | 0.716 | 0.750 | 0 | 0 | 0.000 | 0 |
| nemotron_vl | 175 | 0.797 | 0.923 | 0.942 | 0.918 | 0.905 | 1935 | 18609 | 0.000 | 0 |
| jev_listwise | 175 | 0.632 | 0.923 | 0.948 | 0.817 | 0.837 | 982 | 1421 | 0.580 | 0 |
| jev_pointwise | 175 | 0.597 | 0.890 | 0.932 | 0.781 | 0.805 | 7687 | 9320 | 0.919 | 0 |

_Gold scope: quality metrics over 620 of 700 rows (negative queries have no gold and are excluded, never scored as zero)_

## Per query type

| arm | query type | n | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| embed_only | graphical | 25 | 0.440 | 0.880 | 1.000 | 0.635 | 0.724 |
| embed_only | paraphrase | 40 | 0.600 | 0.900 | 0.925 | 0.734 | 0.782 |
| embed_only | reasoning | 30 | 0.250 | 0.617 | 0.817 | 0.663 | 0.623 |
| embed_only | structural | 20 | 0.550 | 0.850 | 0.900 | 0.677 | 0.732 |
| embed_only | verbatim | 40 | 0.725 | 0.925 | 0.925 | 0.807 | 0.837 |
| jev_listwise | graphical | 25 | 0.800 | 0.960 | 1.000 | 0.887 | 0.915 |
| jev_listwise | paraphrase | 40 | 0.650 | 0.975 | 0.975 | 0.800 | 0.845 |
| jev_listwise | reasoning | 30 | 0.300 | 0.767 | 0.867 | 0.760 | 0.717 |
| jev_listwise | structural | 20 | 0.650 | 0.950 | 0.950 | 0.785 | 0.827 |
| jev_listwise | verbatim | 40 | 0.750 | 0.950 | 0.950 | 0.850 | 0.876 |
| jev_pointwise | graphical | 25 | 0.680 | 0.960 | 0.960 | 0.813 | 0.851 |
| jev_pointwise | paraphrase | 40 | 0.575 | 0.975 | 0.975 | 0.760 | 0.816 |
| jev_pointwise | reasoning | 30 | 0.250 | 0.600 | 0.817 | 0.648 | 0.612 |
| jev_pointwise | structural | 20 | 0.750 | 0.950 | 0.950 | 0.842 | 0.870 |
| jev_pointwise | verbatim | 40 | 0.750 | 0.950 | 0.950 | 0.850 | 0.876 |
| nemotron_vl | graphical | 25 | 0.880 | 1.000 | 1.000 | 0.930 | 0.948 |
| nemotron_vl | paraphrase | 40 | 0.850 | 0.950 | 0.975 | 0.904 | 0.921 |
| nemotron_vl | reasoning | 30 | 0.417 | 0.767 | 0.833 | 0.897 | 0.781 |
| nemotron_vl | structural | 20 | 0.850 | 0.950 | 0.950 | 0.900 | 0.913 |
| nemotron_vl | verbatim | 40 | 0.950 | 0.950 | 0.950 | 0.950 | 0.950 |

## Blind pairwise judging (Jev, position-swapped)

| pairing | A wins | B wins | ties | flips | swap consistency |
|---|---|---|---|---|---|
| jev_listwise>jev_pointwise | 27 | 5 | 143 | 81 | 0.54 |
| nemotron_vl>jev_listwise | 7 | 35 | 133 | 71 | 0.59 |
| nemotron_vl>jev_pointwise | 20 | 17 | 138 | 64 | 0.63 |
| nemotron_vl>embed_only | 55 | 7 | 113 | 66 | 0.62 |
| jev_listwise>embed_only | 69 | 3 | 103 | 51 | 0.71 |
| jev_pointwise>embed_only | 55 | 9 | 111 | 69 | 0.61 |

## Gemini Flash spot check

| pairing | kappa | n | flagged |
|---|---|---|---|
| jev_listwise>jev_pointwise | 0.22 | 50 | 3 |
| nemotron_vl>jev_listwise | 0.17 | 50 | 4 |
| nemotron_vl>jev_pointwise | 0.23 | 50 | 2 |
| nemotron_vl>embed_only | 0.30 | 50 | 0 |
| jev_listwise>embed_only | 0.37 | 50 | 1 |
| jev_pointwise>embed_only | 0.20 | 50 | 4 |
