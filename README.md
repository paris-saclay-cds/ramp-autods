RAMP AutoDS
=============

Automated Tabular Data Scientist based on the RAMP ecosystem.

## Headline results

Across 33 Kaggle tabular challenges (regression and classification, full Kaggle private leaderboard protocol):

| System | mean p-rank | wins (out of 33) |
| --- | --- | --- |
| AutoDS | **81.9** | **18** |
| AutoGluon 1.1.1 (best_quality, matched runtime) | 78.7 | 13 |

p-rank is the percentage rank on the Kaggle private leaderboard, 0 to 100, higher is better. AutoGluon was given the same total wall-clock as AutoDS.

On the small-data subset where TabPFN (Hollmann et al., Nature 2025) can be run, well-hyperopted boosting plus blending wins on 3 of 5 challenges:

| Challenge | TabPFN p-rank | AutoDS blend p-rank |
| --- | --- | --- |
| attrition | 39.3 | **56.6** |
| cirrhosis | 45.6 | **92.0** |
| concrete strength | **99.3** | 94.7 |
| horses | 54.4 | **76.5** |
| wine | **75.9** | 68.6 |

Even single-model AutoDS CatBoost (no blending, no race) is roughly tied with TabPFN on 4 of 5 challenges. See the paper draft (Tables 6 and 7) for full numbers and the apples-to-apples conditions.

## What AutoDS is

A fully automated tabular ML stack with five components, all carefully integrated:

- **Base predictors:** XGBoost, CatBoost, LightGBM, scikit-learn MLP, each with a fixed hyperparameter grid.
- **Hyperopt agent:** HEBO, run as a submission agent so every arm explored becomes a candidate for blending.
- **Partial hyperopt:** optimize predictor or data preprocessor while freezing the other; novel and effective for high-dimensional spaces that include preprocessor hyperparameters.
- **CV-bagging and forward-greedy blending:** 3 by 10-fold CV, blend over folds, blend over submissions per fold.
- **Hyperopt race orchestration agent:** evolutionary-flavored scheduler that allocates compute proportionally to each base predictor's contributivity to the current blend, then refines either the predictor or the preprocessor of a winning submission.

Ablation across these elements (approximate, see paper Table 3):

| element | average p-rank gain ove best single boosting model |
| --- | --- |
| blending | 8 |
| hyperopt and race | 4 |
| optimizing over seeds | 3 |
| early stopping | 2 |
| feature dropout | 2 |
| SKMLP base predictor | 1 |
| missing data imputation | 1 |

Blending dominates. Most of the gap to single-model systems including TabPFN is explained by ensembling well-hyperopted boosters, not by any single architectural choice.

## Why this exists

LLM-based data science agents are improving fast but still rely on an underlying AutoML stack to actually fit and select models. AutoDS is intended as a strong such backbone: a transparent, reproducible, evaluation-honest baseline that outperforms the prior state of the art and is easy to extend. It can be used as:

- a research baseline for new tabular methods, evaluated through the Kaggle private-leaderboard protocol described in the paper;
- the inner loop of an LLM-driven data-science agent, where the LLM does feature engineering and problem framing and AutoDS handles model selection, hyperopt, and blending;
- a working reference implementation of partial hyperopt and the hyperopt race for the AutoML community.

## Installation

```bash
conda create -n auto_ds python=3.11
conda activate auto_ds
pip install .
```

Optional, for end-to-end Kaggle integration:

```bash
pip install ramp-kaggle
```

3. (Optional) To interact with Kaggle you need to install `ramp-kaggle`.

# Setup

1. Clone a RAMP setup kit, for example, [`kaggle_abalone`](https://github.com/ramp-setup-kits/kaggle_abalone), place it in `./ramp-setup-kits` (by convention), and download the data from the kaggle site.
```
ramp-setup-kits
└───kaggle_abalone
    ├───metadata.json
    ├───sample_submission.csv
    ├───test.csv
    └───train.csv
```

2. If you haven't, create the `ramp-kits` folder
```
mkdir ramp-kits
```

3. Set up the first run on your new kit:
```
cd ramp-kits
ramp-setup --ramp-kit kaggle_abalone --version 1_1 --number 1
```
`version` and `number` can be any string. Conventionally we use `version` to mark either a version of `ramp-autods` or a config file specifying command-line parameters, and `number` to mark an execution, like a seed.

If your setup folder is not `../ramp-setup-kits`, you can specify it with `--setup-root`.

The result is a functional RAMP kit (also available [here](https://github.com/ramp-kits/kaggle_abalone), for reference), with the starting kit submission (an LGBM) trained, tested, and scored.

<details><summary>Explanation of the resulting folder structure</summary>

```
ramp-kits
└───kaggle_abalone_v1_1_n1
    ├───actions
    │   ├───'2025-02-24 16:57:58.900437.pkl'
    │   └───...
    ├───data
    │   ├───metadata.json
    │   ├───sample_submission.csv
    │   ├───test.csv
    │   └───train.csv
    ├───problem.py
    └───submissions
        └───starting_kit
            ├───data_preprocessor_0_drop_id.py
            ├───data_preprocessor_1_drop_columns.py
            ├───data_preprocessor_2_col_in_train_only.py
            ├───data_preprocessor_3_Sex_1_cat_col_encoding.py
            ├───data_preprocessor_4_rm_constant_col.py
            ├───feature_extractor.py
            ├───regressor.py
            └───training_output
                └───starting_kit
                    ├───bagged_scores.csv
                    ├───fold_0
                    │   ├───scores.csv
                    │   ├───y_pred_test.npz
                    │   └───y_pred_train.npz
                    ├───...
                    ├───submission_bagged_test.csv 
                    └───submission_bagged_valid.csv
      
```
1. `actions` contains tiny python pickles storing all parameters and return values of function called during the run, marked with the `@ramp_action` decorator in the code. We use this heterogeneous database to recover timings, scores, and other statistics about the run, as well as to resume a crashed run.
2. `data` is a copy of the setup kit, except than `metadata.json` is augmented with stats computed after reading the training and test data.
3. `problem.py` is the [config file that describes the kit](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/problem.html). You can modify it after the setup, if you know what you are doing.
4. `submissions` is the folder with all the submissions (data preprocessors and a predictor, inserted into the [classification]() or [regression]() workflow.). After setup, a single `starting_kit` submission is submitted, containing an LGBM `regressor.py`, and a list of `data_preprocessor`s (some of them fixed, some of them depending on the data). The `feature_extractor.py` is currently blank; data preprocessors are executed once, before the folds are created, while the feature extractor is called for every fold.
5. `submissions/<submission>/training_output` stores all the results, including a table `bagged_scores.csv` for all the foldwise scores and runtimes, and `submission_bagged_test.csv` that is a valid submission file that can be submitted to Kaggle. For each fold, we store `y_pred_train.npz` (training + validation predictions) and `y_pred_test.npz` (test predictions). These can be large files, but we need to keep them until the final blend.
</details>

<details><summary>Optional RAMP test</summary>

The kit is a valid RAMP kit which means that all RAMP commands can be used on it. You can test
```
ramp-test
```
which will re-train the starting kit, but on all 30 CV folds instead of the three folds we use to check in the setup. If you modify `problem.py`, we suggest that you unit test it using `ramp-test`.
</details>
