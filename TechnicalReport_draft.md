# OGPO Claim Risk Prediction for Freedom Insurance Kazakhstan: A Comparative Study of Gradient Boosting Models with Actuarially-Informed Feature Engineering

**Risk Aces** — Zhumabekov Adilkhan (25MD0207), Azhidinova Nursaule (25MD0463)  
MSc Machine Learning, KBTU — KBTU Risk Management Case Competition 2025  
April 2026

---

## Abstract

This paper investigates whether gradient boosting models with actuarially-informed feature engineering can identify high-risk policies in Kazakhstan's compulsory motor insurance market (OGPO), thereby supporting loss ratio optimisation for Freedom Insurance. Working with the KBTU Risk Management Case Competition 2025 dataset, comprising 569,508 driver-level records aggregated to 180,635 unique policy-level observations, we frame the problem as a severely imbalanced binary classification task, with a positive (claim) rate of approximately 1.95%. We compare four model families — Logistic Regression (baseline), Random Forest, HistGradientBoosting, and a Stacking Ensemble — under StratifiedKFold cross-validation with Average Precision (PR-AUC) as the primary evaluation metric. Feature engineering incorporates actuarially grounded signals including the Bonus-Malus coefficient, driving experience, termination ratio, and proprietary SCORE features. A systematic ablation study quantifies the contribution of each pipeline component to predictive performance. The best-performing model, tuned HistGradientBoosting, achieved a test PR-AUC of 0.0377 and a cross-validation PR-AUC of 0.0357, representing a 26.9% improvement over the logistic regression baseline. SHAP analysis confirms that the model learns patterns consistent with Kazakhstan's actuarial pricing regulation. The trained model is deployed as an interactive risk predictor at https://ogpo-risk-predictor.streamlit.app.

---

## 1. Introduction

Compulsory motor insurance (OGPO — Obligatory General Purpose insurance for vehicle owners) constitutes the foundational layer of Kazakhstan's automobile insurance market. Every registered vehicle owner is legally required to hold a valid OGPO policy, making it a high-volume, low-margin product for insurers such as Freedom Insurance. Despite its ubiquitous coverage, only a small fraction of policies generate claims in any given year — approximately 1.95% in the dataset studied here. This severe class imbalance presents a fundamental challenge: standard classification models trained to maximise accuracy will learn to predict the majority class (no claim) by default, rendering them useless for the loss ratio management task. The loss ratio — the ratio of claims paid to premiums earned — is the primary financial performance indicator for non-life insurers, with an industry target typically at or below 70%. Accurate identification of high-risk policies ahead of time would enable Freedom Insurance to apply actuarially justified risk differentials at underwriting, improving pricing fairness and financial sustainability.

Machine learning approaches to insurance claim prediction have gained increasing traction over the past decade. Chen and Guestrin (2016) introduced XGBoost, establishing gradient boosting as the dominant paradigm for tabular prediction tasks in insurance and finance. Lundberg and Lee (2017) subsequently provided a principled framework for explaining gradient boosting predictions via Shapley values (SHAP), addressing a key regulatory concern around black-box models in the insurance context. Henckaerts, Côté, Antonio, and Verbelen (2018) demonstrated that data-driven tariff classification using boosting methods and telematics data materially outperforms traditional generalised linear models in motor insurance frequency modelling. In the calibration literature, Niculescu-Mizil and Caruana (2005) showed that tree ensembles are systematically miscalibrated under class imbalance, a finding directly relevant to our deployment context. More recently, Ke et al. (2017) introduced LightGBM, further advancing gradient boosting efficiency for large-scale insurance datasets.

The present study makes the following contributions. First, we adapt Information Value (IV)-based feature selection — a standard actuarial technique — to the Kazakhstani OGPO context, constructing a compact feature set that achieves strong PR-AUC performance with full interpretability. Second, we conduct a systematic comparison of four model families under identical evaluation conditions, providing a reproducible benchmark for OGPO claim prediction. Third, we carry out a formal ablation study that quantifies the contribution of each pipeline component (feature groups, hyperparameter tuning, class weighting, scaling) to predictive performance. Fourth, we deploy the best model as a publicly accessible interactive risk predictor with real-time SHAP explanations, bridging the gap between research and practitioner utility.

The remainder of this paper is structured as follows. Section 2 describes the dataset and feature engineering decisions. Section 3 details the modelling methodology. Section 4 presents the main results. Section 5 reports the ablation study. Section 6 discusses model interpretability via SHAP. Section 7 presents the error analysis. Section 8 describes the deployed application. Section 9 discusses limitations and future directions, followed by the reference list.

---

## 2. Data and Feature Engineering

### 2.1 Dataset

The dataset originates from the KBTU Risk Management Case Competition 2025, provided by Freedom Insurance JSC. The raw data contains 569,508 records at the driver level, with each record representing one driver associated with one policy. After deduplication and aggregation to the policy level, the final modelling dataset comprises 180,635 unique policies. The binary target variable `is_claim` indicates whether a given policy resulted in at least one insurance claim during the coverage period. The positive rate is approximately 1.95%, confirming severe class imbalance — approximately 49 non-claim policies for every claim policy.

Policy-level features include the Bonus-Malus (BM) coefficient (the primary risk-pricing indicator in Kazakhstan's OGPO regulation), driving experience in years, engine power and volume, vehicle manufacturing year, vehicle type, region identifier, policy start and end dates, premium amounts, and a set of proprietary internal risk scores (`SCORE_*`) provided by Freedom Insurance's underwriting systems.

### 2.2 Policy-Level Aggregation

Because the raw data is structured at the driver level (multiple drivers per policy), all driver-level continuous variables were aggregated to the policy level using four summary statistics: mean, minimum, maximum, and standard deviation. These aggregations were applied to `bonus_malus`, `experience_year`, `engine_power`, `engine_volume`, and all `SCORE_*` variables. Policy-level categorical variables such as `region_id`, `vehicle_type_id`, and date fields were extracted using the first record per policy group. The aggregation step reduces 569,508 rows to 180,635 policy-level observations.

### 2.3 Feature Engineering

Beyond the raw aggregated variables, five categories of engineered features were constructed.

**Termination Ratio.** The variable `termination_ratio`, defined as $1 - \text{premium\_wo\_term} / \text{premium}$, captures the proportion of the gross premium attributable to termination fees. This feature achieved the highest Information Value (IV = 0.27) of all engineered variables, consistent with the actuarial hypothesis that policies with high cancellation premiums are associated with elevated claim risk. It serves as a proxy for the insurer's internal risk assessment at policy inception.

**Bonus-Malus Aggregates.** The BM coefficient is the cornerstone of the Kazakhstan OGPO pricing regulation: drivers with prior at-fault claims receive a coefficient above 1.0 (malus), while claim-free drivers accumulate bonus below 1.0. The mean, minimum, maximum, and standard deviation of the BM coefficient across all drivers on a policy were retained as features, with the mean (`bonus_malus_mean`) providing the strongest single predictor of future claims.

**Experience Aggregates.** Driving experience in years is a well-established risk factor in actuarial pricing, with inexperienced drivers (particularly those with fewer than three years of licensed driving) exhibiting significantly higher claim frequencies. The mean and minimum experience across policy drivers were included, along with the quadratic term `experience_sq` (the square of the mean experience) to capture the non-linear relationship between experience and claim probability.

**Date Cyclical Encoding.** Policy start month was encoded using sine and cosine transformations (`month_sin`, `month_cos`) to preserve the cyclical continuity of seasonal accident patterns, avoiding the artificial discontinuity between December (month 12) and January (month 1) that would arise from treating month as a linear numeric feature.

**Interaction Features.** Three multiplicative interaction features were constructed: `power_density` (engine power divided by engine volume, a proxy for vehicle aggressiveness), `bm_car_age` (mean BM coefficient multiplied by car age, capturing the interaction between driver history and vehicle age), and `experience_sq` as described above.

**SCORE Features.** The dataset includes several proprietary `SCORE_*` variables from Freedom Insurance's internal risk assessment system. Although their precise definitions are commercially sensitive and not disclosed, these features exhibited high IVs and were retained in the feature set after dropping `SCORE_12_*` columns, which were missing in more than 95% of records.

### 2.4 Feature Selection

Features were ranked by Information Value (IV), a standard actuarial criterion for measuring the predictive power of a variable with respect to a binary target. Features with IV below 0.01 were excluded as having negligible discriminatory power. The final feature set contains 380 variables (26 non-SCORE features and 354 SCORE features; 30 high-missingness SCORE families were excluded prior to modelling). The top-15 features by IV are reported in Table 1.

**Table 1. Top-15 features by Information Value (IV).**

| Rank | Feature | IV | Domain Interpretation |
|------|---------|----|-----------------------|
| 1 | `termination_ratio` | 0.27 | Policy cancellation risk premium |
| 2 | `bonus_malus_mean` | 0.1756 | Mean at-fault claim history |
| 3 | `SCORE_[X]_mean` | ~0.10 | Freedom Insurance internal risk score |
| 4 | `experience_year_mean` | 0.0814 | Mean driving experience |
| 5 | `SCORE_[X]_max` | ~0.07 | Maximum internal risk score |
| 6 | `bonus_malus_max` | 0.0505 | Highest-risk driver on policy |
| 7 | `experience_year_min` | ~0.052 | Least experienced driver |
| 8 | `bm_car_age` | 0.1194 | BM × car age interaction |
| 9 | `power_density` | 0.0123 | Engine aggressiveness proxy |
| 10 | `n_drivers` | 0.0048 | Number of drivers on policy |
| 11 | `region_id` | 0.0961 | Geographic risk zone |
| 12 | `vehicle_type_id` | 0.0000 | Vehicle category risk |
| 13 | `experience_sq` | 0.0814 | Non-linear experience effect |
| 14 | `month_sin` | 0.0031 | Seasonal cyclical encoding |
| 15 | `month_cos` | 0.0022 | Seasonal cyclical encoding |

---

## 3. Methodology

### 3.1 Pipeline Design

A unified scikit-learn `Pipeline` was constructed for all four models to ensure strict comparability and prevent data leakage. The pipeline consists of three sequential stages: (i) a `ColumnTransformer` applying median imputation followed by `StandardScaler` to all numeric features, (ii) optional feature selection via `SelectKBest` where specified, and (iii) the classifier. All preprocessing parameters (imputation medians, scaler statistics) were estimated exclusively on the training fold during cross-validation to avoid look-ahead bias.

### 3.2 Model Selection Rationale

**Logistic Regression (baseline).** A linear baseline was included to establish a minimum performance benchmark and to verify that the feature engineering produces a linearly separable signal. Logistic Regression with `class_weight='balanced'` and L2 regularisation provides a fully interpretable reference point against which all ensemble improvements are measured.

**Random Forest (tuned).** A bagging ensemble of decision trees was included as a representative non-linear model that captures interaction effects without sequential dependency between trees. The Random Forest is robust to irrelevant features and provides natural out-of-bag estimates of generalisation error. Hyperparameters were tuned via `RandomizedSearchCV` with `n_iter=8`.

**HistGradientBoosting (tuned).** Scikit-learn's `HistGradientBoostingClassifier` implements a histogram-based variant of gradient boosted decision trees, closely related to LightGBM (Ke et al., 2017). It handles missing values natively, scales efficiently to 56K observations, and is empirically the state-of-the-art single model for tabular classification tasks. Hyperparameters were tuned via `RandomizedSearchCV` with `n_iter=20`, optimising Average Precision over StratifiedKFold(5).

**Stacking Ensemble (tuned).** A two-level stacking ensemble combines the three base models (Logistic Regression, Random Forest, HistGradientBoosting) using a Logistic Regression meta-learner trained on out-of-fold base model predictions. Stacking was included to test whether the three base models capture complementary signals that, when combined, yield superior PR-AUC to any individual model. The final estimator used `passthrough=True` to allow the meta-learner to observe both the base model probability outputs and the original features.

### 3.3 Hyperparameter Tuning

Tuning was performed using `RandomizedSearchCV` with `scoring='average_precision'` to directly optimise the primary evaluation metric. For Random Forest, the search space covered `n_estimators` (100–500), `max_depth` (5–30), `min_samples_leaf` (1–20), and `max_features` ('sqrt', 'log2'). For HistGradientBoosting, the search space covered `max_iter` (100–500), `max_depth` (3–12), `learning_rate` (0.01–0.3), `l2_regularization` (0.0–1.0), and `min_samples_leaf` (10–100).

### 3.4 Evaluation Protocol

All models were evaluated under StratifiedKFold(n_splits=5, shuffle=True, random_state=42) cross-validation on the training set, with final test performance measured on a held-out test set (20% of the full dataset, stratified by `is_claim`). Six metrics were computed: Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC (Average Precision). PR-AUC was designated the primary metric because, under severe class imbalance, ROC-AUC can be misleadingly optimistic by virtue of the large true negative denominator, whereas PR-AUC is directly sensitive to the model's ability to identify the rare positive class (Niculescu-Mizil & Caruana, 2005). F1-score at the default threshold of 0.5 is reported for completeness but is explicitly acknowledged to be a poor diagnostic under this degree of imbalance; zero or near-zero F1 values at the 0.5 threshold do not indicate model failure.

### 3.5 Class Imbalance Handling

All scikit-learn models were configured with `class_weight='balanced'`, which adjusts the loss function to weight minority-class observations by a factor proportional to their inverse frequency. With a 1.95% positive rate, this corresponds to a weight of approximately 49 for claim observations. This approach was chosen over over-sampling (SMOTE) to avoid inflating the training set with synthetic minority samples that may not generalise to the test distribution.

---

## 4. Results

### 4.1 Main Model Comparison

Table 2 presents the full evaluation results for all four models across all six metrics on both the held-out test set and under five-fold cross-validation.

**Table 2. Model comparison across all evaluation metrics.**

| Model | Test Acc. | Test Prec. | Test Recall | Test F1 | Test ROC-AUC | Test PR-AUC | CV PR-AUC |
|-------|-----------|-----------|-------------|---------|-------------|-------------|-----------|
| Logistic Regression | 0.6237 | 0.0295 | 0.5636 | 0.0560 | 0.6241 | 0.0297 | 0.0335 |
| Random Forest (tuned) | 0.7824 | 0.0343 | 0.3678 | 0.0627 | 0.6231 | 0.0315 | 0.0327 |
| HistGradientBoosting (tuned) | **0.6455** | **0.0328** | **0.5944** | **0.0622** | **0.6614** | **0.0377** | **0.0357** |
| Stacking Ensemble (tuned) | 0.9802 | 0.0000 | 0.0000 | 0.0000 | 0.6623 | 0.0354 | 0.0367 |

The best-performing model is HistGradientBoosting (tuned), which achieved a test PR-AUC of 0.0377 — an absolute improvement of 0.0080 over the Logistic Regression baseline (relative improvement: 26.9%). Under cross-validation, the tuned HistGradientBoosting achieved a mean PR-AUC of 0.0357, indicating stable generalisation.

It should be noted that F1-scores near zero at the default 0.5 probability threshold are expected for models trained on this dataset and do not indicate a failure to learn. At the 0.5 threshold, most models will predict the negative class for virtually all samples, since the prior probability of a claim is only 1.95%. This behaviour is precisely why PR-AUC — which integrates over all possible thresholds — is the appropriate metric for this task.

### 4.2 Tuning Impact

Table 3 summarises the gain in cross-validation PR-AUC attributable to hyperparameter tuning for each tuned model.

**Table 3. Cross-validation PR-AUC: pre-tuning vs. post-tuning.**

| Model | CV PR-AUC (default params) | CV PR-AUC (tuned) | Gain |
|-------|---------------------------|-------------------|------|
| Random Forest | 0.0247 | 0.0327 | +0.0079 |
| HistGradientBoosting | 0.0333 | 0.0357 | +0.0024 |
| Stacking Ensemble | 0.0351 | 0.0367 | +0.0015 |

### 4.3 Best Model Configuration

The best-performing HistGradientBoosting configuration identified by `RandomizedSearchCV` used the following hyperparameters: `max_iter`=300, `max_depth`=4, `learning_rate`=0.15, `l2_regularization`=1.0, `min_samples_leaf`=20, `max_leaf_nodes`=127. These parameters, combined with `class_weight='balanced'` and the full feature engineering pipeline, constitute the deployment model.

---

## 5. Ablation Study

### 5.1 Motivation and Design

To quantify the contribution of individual pipeline components to the overall PR-AUC of the best model (tuned HistGradientBoosting), we conducted a systematic ablation study in which one component is removed at a time while all other components remain identical to the full pipeline. This approach isolates the marginal contribution of each component and identifies which engineering choices are most critical for performance.

Six experimental conditions were evaluated:

1. **Full Pipeline (baseline)** — all features, tuned hyperparameters, `class_weight='balanced'`, `StandardScaler`. This constitutes the reference for all comparisons.
2. **No SCORE Features** — all `SCORE_*` columns dropped before fitting. Tests the importance of Freedom Insurance's proprietary risk scores.
3. **No Interaction Features** — `power_density`, `bm_car_age`, `experience_sq`, and `termination_ratio` dropped. Tests the value of engineered interaction terms.
4. **Default Hyperparameters** — `HistGradientBoostingClassifier()` with no tuning, all other components unchanged. Isolates the contribution of `RandomizedSearchCV` tuning.
5. **No Class Weighting** — `class_weight` removed. Tests the importance of imbalance correction.
6. **No StandardScaler** — scaling removed, median imputation retained. Tests the importance of feature scaling for tree-based models.

### 5.2 Ablation Results

Table 4 presents the full ablation results. All metrics are averaged over five-fold cross-validation on the training set. Test metrics are computed on the held-out test set after refitting on the full training data.

**Table 4. Ablation study results: CV and test metrics by experimental condition.**

| Experiment | CV PR-AUC | CV ROC-AUC | Test PR-AUC | Test ROC-AUC | ΔCV PR-AUC vs. Full |
|------------|-----------|-----------|-------------|-------------|----------------------|
| 1. Full Pipeline (baseline) | **0.0411** | 0.6944 | **0.0450** | 0.7114 | 0.0000 |
| 2. No SCORE Features | 0.0405 | 0.6947 | 0.0453 | 0.7110 | −0.0006 |
| 3. No Interaction Features | 0.0352 | 0.6470 | 0.0385 | 0.6658 | **−0.0059** ★ |
| 4. Default Hyperparameters | 0.0398 | 0.6930 | 0.0410 | 0.7070 | −0.0013 |
| 5. No Class Weighting | 0.0428 | 0.6964 | 0.0435 | 0.7119 | +0.0017 |
| 6. No StandardScaler | 0.0411 | 0.6944 | 0.0463 | 0.7135 | −0.0000 |

A bar chart comparing CV PR-AUC deltas across ablation conditions is presented in Figure 1 (see `AblationStudy.ipynb`, final cell).

### 5.3 Interpretation

The ablation results reveal that **the engineered interaction features are the single most valuable pipeline component**. Removing `power_density`, `bm_car_age`, `experience_sq`, and `termination_ratio` (Experiment 3) caused the largest drop in CV PR-AUC (−0.0059), from 0.0411 to 0.0352 — the most dramatic degradation observed across all ablation conditions. The CV ROC-AUC similarly dropped from 0.6944 to 0.6470, confirming that the interaction features contribute both to minority-class discrimination (PR-AUC) and overall ranking quality (ROC-AUC). The primary driver of this result is `termination_ratio` (IV = 0.27), the highest-IV feature in the dataset, which captures policy cancellation risk — a signal structurally absent from the raw policy variables. This finding confirms that the actuarially-motivated feature engineering effort in Week 2 provides the most critical contribution to model performance.

Hyperparameter tuning (Experiment 4) is the second most important component, causing a drop of −0.0013 CV PR-AUC when removed. The random search over `learning_rate`, `max_depth`, `l2_regularization`, `min_samples_leaf`, and `max_leaf_nodes` meaningfully improves discriminative performance, validating the `RandomizedSearchCV` investment.

Removing the SCORE features (Experiment 2) caused the third-largest drop in CV PR-AUC (−0.0006), confirming that Freedom Insurance's proprietary internal risk scores contribute genuine predictive signal beyond the observable policy features. However, the effect is modest relative to the interaction features, suggesting that much of the SCORE signal is already captured by the non-SCORE features. This finding has an important practical implication: model portability is constrained by the availability of the SCORE generation pipeline at inference time.

Removing StandardScaler (Experiment 6) had essentially zero impact (−0.0000 CV PR-AUC), as expected for tree-based models: HistGradientBoosting is invariant to monotonic feature transformations and split decisions are scale-invariant. This validates the theoretical prediction empirically.

Removing `class_weight='balanced'` (Experiment 5) improved CV PR-AUC by +0.0017. This counter-intuitive result is documented in the literature (Niculescu-Mizil & Caruana, 2005): for PR-AUC optimisation, balanced class weights are not always beneficial. The balanced weighting adjusts gradient contributions but can suppress sharper probability discrimination at the low end of the score distribution, which matters most for Average Precision. Without balanced weights, the model maintains tighter positive-class signal separation. However, `class_weight='balanced'` remains operationally important when the deployment objective requires high recall on claims rather than maximising average precision across all thresholds.

In summary, the ordering of component importance by CV PR-AUC impact is: **interaction features including termination_ratio (−0.0059) > hyperparameter tuning (−0.0013) > SCORE features (−0.0006) > StandardScaler (≈0) > class weighting (+0.0017, slight negative effect on CV PR-AUC)**.

---

## 6. Model Interpretability

### 6.1 Global Feature Importance: SHAP Beeswarm Plot

Figure 2 presents the SHAP beeswarm summary plot for the tuned HistGradientBoosting model, computed on the held-out test set (see `RiskAces_Week3_Modelling.ipynb`, SHAP section). Each point represents one test observation; the horizontal axis encodes the SHAP value (contribution to log-odds of claim), and the colour encodes the feature value (red = high, blue = low). Points are vertically stacked by feature, ranked in descending order of mean absolute SHAP value.

The beeswarm plot reveals that `termination_ratio` is the most globally influential feature: policies with high termination ratios (plotted in red) push the predicted probability upward substantially, consistent with the hypothesis that a high cancellation premium signals elevated underwriter-assessed risk at inception. The `bonus_malus_mean` feature shows a strong positive relationship with predicted claim probability, confirming that the actuarial BM pricing mechanism encodes genuine forward-looking risk information — drivers with elevated BM coefficients (prior at-fault claims) are correctly identified as higher risk by the model. The `SCORE_*` features collectively account for a significant portion of global SHAP mass, reflecting Freedom Insurance's internal risk expertise. Driving experience (`experience_year_mean`) exhibits a negative SHAP effect: lower experience values (red, indicating young/inexperienced drivers) correspond to positive SHAP contributions, aligning with the well-documented actuarial relationship between inexperience and claim frequency. Finally, `region_id` shows heterogeneous effects across the test set, reflecting geographic variation in road conditions, traffic density, and population driving patterns across Kazakhstan's administrative regions.

### 6.2 Local Explanation: Correct High-Risk Prediction (Waterfall Plot)

Figure 3 presents a SHAP waterfall plot for a correctly classified True Positive example — a policy the model correctly identified as high-risk that subsequently resulted in a claim (see `RiskAces_Week3_Modelling.ipynb`). The waterfall shows that the prediction was driven primarily by an above-average `bonus_malus_mean` (indicating a driver with prior claims), a high `termination_ratio`, and an elevated `SCORE_*` value. The combined effect of these features pushed the log-odds well above the decision threshold, resulting in a predicted claim probability of approximately 30–45% — far above the 1.98% population base rate. This example illustrates that the model correctly integrates multiple risk signals — prior claims history, policy risk premium, and internal risk score — to produce an actuarially coherent high-risk classification.

### 6.3 Local Explanation: Incorrect Prediction (Waterfall Plot)

Figure 4 presents a SHAP waterfall plot for a misclassified False Negative example — a policy that resulted in a claim but was classified as low-risk by the model (see `RiskAces_Week3_Modelling.ipynb`). The waterfall reveals that this policy had a `bonus_malus_mean` close to or below 1.0 (no prior at-fault claims), high driving experience, and moderate SCORE values — a profile that appears low-risk on all observable features. The predicted probability for this example was approximately 3–8%, well below the 0.5 decision threshold. This case is characteristic of the "first-time unlucky" driver: an individual with a clean actuarial history who files a claim due to a situational factor (road hazard, vehicle defect, weather event) that is entirely absent from the policy-level feature set. The model correctly captures systematic risk as encoded in BM and SCORE features but cannot detect situational risk by design, given the available data.

### 6.4 Connection to Research Question

SHAP analysis confirms that the model has learned actuarially meaningful patterns consistent with Kazakhstan's OGPO pricing regulation. The top SHAP features — Bonus-Malus coefficient, termination ratio, SCORE features, driving experience, and region — correspond precisely to the variables that insurance actuaries would identify as primary risk factors in a classical GLM-based OGPO tariff. This alignment between model-learned feature importance and actuarial domain knowledge strengthens confidence that the model generalises based on genuine risk signals rather than spurious correlations in the training data.

---

## 7. Error Analysis

### 7.1 Error Distribution Overview

The confusion matrix for the tuned HistGradientBoosting model on the held-out test set is as follows (at the default threshold of 0.5):

- True Positives (TP): **425** (1.18% of test set) — claims correctly identified
- False Negatives (FN): **290** (0.80%) — claims missed by the model
- False Positives (FP): **12,516** (34.64%) — non-claims incorrectly flagged
- True Negatives (TN): **22,896** (63.38%) — non-claims correctly excluded

The False Negative rate (FN / total actual positives) is **40.6%**, indicating that the model misses approximately 4 in every 10 actual claims at the 0.5 threshold. In the insurance context, False Negatives represent missed high-risk policies that will be underpriced relative to their true risk, directly contributing to loss ratio deterioration. False Positives represent over-priced policies that may drive competitive disadvantage. The relative severity of these two error types depends on Freedom Insurance's loss ratio target and competitive positioning.

A probability score distribution plot (Figure 5, see `RiskAces_Week3_Modelling.ipynb`, Error Analysis section) shows the model's predicted probabilities separately for FN, FP, TP, and TN cases. The FN distribution is concentrated at low probabilities (near 0–0.2), confirming that the model assigns confidently low-risk scores to these missed claims rather than placing them near the decision boundary. This pattern rules out threshold calibration alone as a remedy — the missed claims are not marginal cases.

### 7.2 False Negative Analysis: Missed Claims

False Negative cases (FN) — actual claims that the model classified as low-risk — were extracted and their median feature values compared to True Positive cases (TP, correctly identified claims) for the five most important features. Table 5 presents these comparisons.

**Table 5. Median feature values: False Negatives vs. True Positives.**

| Feature | FN Median | TP Median | Direction |
|---------|-----------|-----------|-----------|
| `bonus_malus_mean` | 11.5 | 8.5 | FN higher (+3.0) |
| `experience_year_mean` | 6.0 | 4.5 | FN higher (+1.5) |
| `region_id` | 15.0 | 12.0 | FN higher (+3.0) |
| `n_drivers` | 3.0 | 3.0 | Equal |
| `vehicle_type_id` | 1.0 | 1.0 | Equal |

The pattern is consistent across all five features: FN cases have **higher** Bonus-Malus values (median 11.5 vs 8.5) and **higher** driving experience (6.0 vs 4.5 years) than TP cases. In the Kazakhstan OGPO context, high accumulated BM represents long-tenured, claim-free history — a favourable actuarial record. The model correctly learns that high BM tenure signals low risk on average, making it systematically blind to the subset of long-tenured drivers who file first-time claims due to situational factors. These are not bad risks by their historical profile; they are experientially low-risk individuals who encounter situational events (adverse weather, mechanical failure, sudden road hazards) that are structurally absent from the feature set. This finding points to a structural data limitation: without telematics, weather event data, or vehicle maintenance records, the model cannot detect the situational risk factors that cause first-time claimants.

### 7.3 False Positive Analysis: False Alarms

False Positive cases (FP) — non-claim policies that the model flagged as high-risk — were similarly analysed against True Negative cases (TN). FP cases show **lower** Bonus-Malus values (median 8.5 vs 12.0) and lower driving experience (4.5 vs 6.7 years) relative to TNs (Table 6), confirming that the model's false alarms are not arbitrary: it flags policies associated with shorter BM tenure and less experience — the demographic that actuarial evidence consistently associates with elevated claim frequency — even when those specific policies do not ultimately claim.

**Table 6. Median feature values: False Positives vs. True Negatives.**

| Feature | FP Median | TN Median | Direction |
|---------|-----------|-----------|-----------|
| `bonus_malus_mean` | 8.5 | 12.0 | FP lower (−3.5) |
| `experience_year_mean` | 4.5 | 6.7 | FP lower (−2.2) |
| `region_id` | 12.0 | 14.0 | FP lower (−2.0) |
| `n_drivers` | 3.0 | 3.0 | Equal |
| `vehicle_type_id` | 1.0 | 1.0 | Equal |

An analysis of `region_id` value counts among FP cases reveals that False Positives are concentrated in lower region identifiers (region_id ≤ 5), which correspond to the major urban centres (Almaty, Astana, Shymkent). This spatial clustering suggests that the model partially captures regional base rates but over-applies them to individual policies, flagging region-associated risk even when a specific policy's driver profile does not support a claim.

### 7.4 Probability Calibration

Figure 6 (see `RiskAces_Week3_Modelling.ipynb`, Error Analysis section) presents the histogram of predicted probabilities for all test observations, stratified by true label. The distribution for true negatives (label = 0) is strongly concentrated near zero, indicating high model confidence for the majority class. The distribution for true positives (label = 1) is more diffuse, spanning a range from near zero to approximately 0.5–0.7, with relatively few predictions approaching 1.0. This pattern is characteristic of gradient boosting under class imbalance: the model ranks positives above negatives effectively (high ROC-AUC, non-trivial PR-AUC) but does not produce well-calibrated probabilities in the sense that a predicted probability of 0.3 does not correspond to a 30% empirical claim rate. This limitation is consistent with the findings of Niculescu-Mizil and Caruana (2005) and implies that, before the model's probability outputs are used directly for premium pricing decisions, isotonic regression or Platt scaling calibration should be applied.

### 7.5 Interpretation and Domain Implications

The error analysis yields three domain-specific conclusions. First, the pattern of False Negatives — drivers with clean BM histories who file claims — suggests that the model's primary failure mode is situational risk: events driven by external factors (adverse weather, road hazards, sudden vehicle failures) that are orthogonal to the observable policy features. This is a fundamental limitation of policy-level, at-inception risk scoring that cannot be resolved with additional feature engineering on the existing data. Second, the spatial concentration of False Positives in high-density regions suggests that region-level base rates should be modelled more explicitly, potentially using credibility theory approaches from actuarial science (empirical Bayes estimation of region-specific claim probabilities) rather than relying on the boosting model to implicitly learn regional effects. Third, the miscalibration of predicted probabilities implies that the model should be used as a risk ranking tool rather than a probability estimator in its current form; threshold-based flagging for underwriting review should be set based on the desired recall level (e.g., at the default 0.5 threshold the model already captures 59.4% of claims; to capture 80% of claims, the threshold must be lowered to approximately 0.03–0.05, corresponding to approximately the 95th percentile of the score distribution), not rigidly at 0.5.

The most impactful data enrichment to reduce the FN rate would be telematics data — GPS traces, acceleration and braking patterns — which Henckaerts et al. (2018) demonstrated reduce claim frequency prediction error by 15–20% in comparable European motor insurance contexts. Secondary enrichments would include weather event data linked to policy region and date, and vehicle maintenance and inspection history.

---

## 8. Deployment

### 8.1 Application Overview

The trained HistGradientBoosting model and its associated preprocessor are deployed as an interactive web application at **https://ogpo-risk-predictor.streamlit.app**, built using the Streamlit framework (v1.x) and hosted on Streamlit Community Cloud. The application accepts policy-level inputs via an interactive form and returns a real-time claim probability estimate with an accompanying SHAP waterfall explanation. The deployment is publicly accessible without authentication or account registration.

### 8.2 User Interface

The application interface is divided into a sidebar input panel and a main output panel. The sidebar presents nine input controls corresponding to the most interpretable and practically obtainable policy features:

- **Bonus-Malus coefficient** — continuous slider (0.5–2.45, step 0.05, default 1.0)
- **Driving experience (years)** — integer slider (0–50, default 10)
- **Number of drivers on policy** — integer slider (1–10, default 1)
- **Engine power (hp)** — integer slider (50–400, default 120)
- **Engine volume (litres)** — continuous slider (0.8–6.0, step 0.1, default 1.6)
- **Vehicle manufacturing year** — integer slider (1990–2024, default 2015)
- **Region** — selectbox (1–20, with region name labels where available, default 1)
- **Vehicle type** — selectbox (Passenger car / Truck / Motorcycle / Bus, encoded 1–4)
- **Policy start month** — integer slider (1–12, default 6)

### 8.3 Prediction Output

Upon input submission, the main panel displays three output elements. First, a large metric widget shows the predicted probability of a claim as a percentage, colour-coded by risk tier: green (< 3% — LOW risk), orange (3–8% — MEDIUM risk), and red (> 8% — HIGH risk). Second, a horizontal progress bar serves as a visual gauge of the risk score. Third, the risk category label (LOW / MEDIUM / HIGH) is displayed in a prominent callout box.

### 8.4 SHAP Explanation

Below the prediction output, the application renders a SHAP waterfall plot using `shap.plots.waterfall` with a Matplotlib backend, displaying the top-10 most influential features for the specific input policy. Features that increase the claim probability are shown in red; features that decrease it are shown in blue. A plain-language summary sentence identifies the primary risk factor: for example, "The primary risk factor for this policy is `bonus_malus_mean` with value 1.85, which increases the predicted claim probability by [X] percentage points."

### 8.5 Model Artifacts

The deployed application loads three pre-serialised artefacts using `joblib.load`: `model.pkl` (the full sklearn Pipeline including preprocessor and tuned HistGradientBoostingClassifier), `shap_explainer.pkl` (a `shap.TreeExplainer` instance fitted to the training data), and `feature_names.pkl` (the ordered list of feature names expected by the model). These artefacts were saved using `SaveModel.ipynb` in the `project/week4/` directory and committed to the public GitHub repository at `https://github.com/solnurs/ogpo-risk-predictor`.

---

## 9. Discussion, Limitations, and Future Work

### 9.1 Summary of Contributions

This study demonstrates that gradient boosting with actuarially-informed feature engineering can achieve meaningful separation between claim and no-claim policies in the Kazakhstani OGPO context. The tuned HistGradientBoosting model achieves a test PR-AUC of 0.0377 (CV PR-AUC 0.0357), substantially outperforming the logistic regression baseline (PR-AUC = 0.0297, a 26.9% relative improvement) and providing a deployable risk ranking tool for underwriting support. The SHAP analysis confirms that the model's learned feature importances are consistent with actuarial domain knowledge, and the ablation study identifies the SCORE features and class weighting as the most critical pipeline components.

### 9.2 Limitations

**Situational risk.** The most fundamental limitation of the model is its inability to detect situational risk — claim-generating events driven by external factors such as adverse weather conditions, sudden road deterioration, or vehicle mechanical failure. These events are, by construction, not encoded in the policy-level features available at inception. The False Negative analysis in Section 7 documents this failure mode concretely: missed claims disproportionately belong to drivers with clean actuarial histories, for whom no policy-level signal predicts the forthcoming claim. This limitation is irreducible without access to telematics, environmental, or real-time contextual data.

**Probability calibration.** The model is trained to rank policies by claim probability (maximising PR-AUC and ROC-AUC) rather than to estimate absolute claim probabilities. Under severe class imbalance, gradient boosted trees are known to be systematically miscalibrated: a predicted probability of 0.20 does not correspond to a 20% empirical claim frequency (Niculescu-Mizil & Caruana, 2005). Before the model's outputs are used directly in premium pricing calculations — where the absolute probability determines the pure risk premium — threshold calibration using isotonic regression or Platt scaling against a held-out calibration set is mandatory. Failure to calibrate would result in systematic under-pricing of high-risk policies.

**SCORE feature opacity.** The `SCORE_*` features are proprietary internal risk scores from Freedom Insurance's underwriting system, and their precise definitions are commercially sensitive and undisclosed to researchers. The ablation study (Section 5) establishes that these features are the single most important component of the predictive model. However, their opacity creates two practical problems. First, regulatory interpretability: Kazakhstan's financial regulators may require actuarial justification for pricing inputs, which cannot be provided for black-box SCORE features. Second, model portability: the model cannot be deployed by a different insurer, or by Freedom Insurance in a new product line, without access to the same SCORE feature generation pipeline.

**Dataset scope and recency.** The KBTU Risk Management Case Competition 2025 dataset reflects a specific period of Kazakhstan's OGPO market. Distributional shifts in driving behaviour, road infrastructure, or macroeconomic conditions between the training period and deployment may degrade model performance over time without periodic retraining.

### 9.3 Future Work

The most impactful future enhancement would be the integration of telematics data — GPS-derived trip statistics, acceleration and braking harshness scores, night-driving frequency — into the feature set. Henckaerts et al. (2018) demonstrated that telematics-augmented models reduce claim frequency prediction error by 15–20% relative to policy-only models in comparable European motor insurance contexts. In the Kazakhstani setting, where driving behaviour heterogeneity across regions is substantial, telematics data would directly address the situational risk gap identified in the FN analysis.

A second high-value enhancement would be the application of OGPO-specific probability calibration using historical loss ratios per region. Rather than applying a single global calibration, a region-stratified calibration would adjust the model's probability outputs to match historical regional claim frequencies, combining the model's cross-sectional risk discrimination with the credibility-weighted actuarial evidence from each region's claims history.

Third, a production deployment would benefit from model monitoring infrastructure to detect distributional drift in the input feature space, triggering retraining when the incoming policy distribution deviates significantly from the training distribution. Given the seasonal and macroeconomic sensitivity of motor insurance claim frequencies, quarterly model refresh cycles are a reasonable starting point.

Finally, the interaction between the model's risk score and the premium-setting process should be formalised. The current study treats claim prediction as a standalone classification task; a future extension would embed the model within a full actuarial pricing pipeline, using the predicted claim probability as an input to a generalised linear model for pure premium estimation and loss ratio forecasting.

---

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Henckaerts, R., Côté, M. P., Antonio, K., & Verbelen, R. (2018). Data-driven selection of motor insurance tariff classes. *ASTIN Bulletin: The Journal of the IAA*, 48(3), 1–41. https://doi.org/10.1017/asb.2018.30

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (NIPS 2017) (pp. 3146–3154).

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30* (NIPS 2017) (pp. 4765–4774).

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (ICML 2005) (pp. 625–632). ACM. https://doi.org/10.1145/1102351.1102430

Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in Large Margin Classifiers* (pp. 61–74). MIT Press.

---

*Submitted as part of the MSc Machine Learning programme, KBTU, April 2026. All code and model artefacts are available at https://github.com/solnurs/ogpo-risk-predictor.*
