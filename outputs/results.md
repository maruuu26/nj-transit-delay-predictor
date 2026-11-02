# Results

Target: a stop is "late" if the train is 5+ minutes behind schedule.
Training data: 787,250 stops before 2019-05-01. Threshold chosen on 108,926 stops from 2019-05-01 to 2019-07-01. Test: 326,252 stops from 2019-07-01 to the end of 2019.
Share of late stops in the test period: 30.7%.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Always predict on time | 0.000 | 0.000 | 0.000 | 0.500 | 0.307 |
| Logistic regression (baseline) | 0.350 | 0.843 | 0.494 | 0.620 | 0.397 |
| Gradient boosting | 0.369 | 0.845 | 0.514 | 0.664 | 0.459 |

PR-AUC of a model with no skill equals the late rate above. Rows use only schedule-known features, so these numbers show how much delay risk is predictable before a train runs.
