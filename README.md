# Feature Interactions Under Distribution Shift

## An Exploratory Study of Simultaneous Feature Shifts and Model Performance

### Overview

This project investigates how machine-learning model performance changes when multiple input features undergo distribution shifts simultaneously.

The main focus is to determine whether the performance degradation caused by simultaneous shifts is simply the sum of the individual effects of each feature shift, or whether interactions between shifting features lead to a different outcome.

The study uses the **UCI Bike Sharing Dataset** and examines environmental features such as temperature, humidity, and windspeed.

---

## Research Question

> **When multiple features shift simultaneously, is the resulting model-performance degradation simply the sum of their individual effects, or do interactions between shifting features change the outcome?**

---

## Objectives

The main objectives of this study are:

1. Establish a baseline machine-learning model and measure its performance.
2. Apply controlled distribution shifts to individual features.
3. Apply shifts to pairs of features simultaneously.
4. Compare the observed combined effect with the sum of the individual effects.
5. Measure the resulting interaction effect.
6. Test whether the observed pattern remains consistent across different random seeds.
7. Perform an additional boundary-safe sensitivity analysis.
8. Compare the main finding with a Random Forest model as a robustness check.

---

## Dataset

The study uses the **Bike Sharing Dataset** from the **UCI Machine Learning Repository**.

The analysis uses:

```text
hour.csv
```

The dataset contains hourly bike-sharing rental information together with weather and calendar-related variables.

The dataset contains:

- **17,379 observations**
- **17 columns**
- **No missing values**

### Target Variable

The prediction target is:

```text
cnt
```

which represents the total number of bike rentals.

### Input Features

The baseline model uses:

```text
temp
hum
windspeed
hr
season
workingday
weathersit
```

The following variables were excluded from the baseline input:

- `instant` — record identifier
- `dteday` — raw date field
- `casual` — component of the target
- `registered` — component of the target
- `cnt` — target variable

`casual` and `registered` were excluded because they are components of `cnt` and would introduce target leakage.

---

## Methodology

The overall workflow of the study is:

```text
Dataset
   ↓
Data Exploration
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Baseline Model
   ↓
Individual Feature Shifts
   ↓
Simultaneous Feature Shifts
   ↓
Interaction Calculation
   ↓
Robustness Checks
   ↓
Sensitivity Analysis
   ↓
Interpretation
```

### Train/Test Split

The dataset is divided into training and testing sets using:

- Test size: **20%**
- Random state: **42**

The model is trained only on the original training data.

The distribution shifts are then applied to the **test data only**.

This allows the experiment to simulate a situation where the data distribution changes after a model has already been trained.

---

## Baseline Model

The primary model is **Linear Regression**.

The model is trained using:

```text
temp
hum
windspeed
hr
season
workingday
weathersit
```

The baseline performance is evaluated using:

- Mean Absolute Error (MAE)
- R² score

### Baseline Performance

The baseline Linear Regression model achieved:

```text
MAE = 106.961
R²  = 0.343
```

The model is not intended to be a state-of-the-art predictor of bike demand. Instead, it provides a controlled baseline for studying the effects of distribution shift.

---

## Distribution Shift

The primary experiment applies a controlled positive shift of:

```text
+0.10
```

to selected environmental features in the **test set**.

The shifted value is calculated using:

```python
x_shifted = (x + 0.10).clip(0, 1)
```

The trained model is **not retrained** after the shift.

This is important because the experiment is designed to measure how a previously trained model responds when the input distribution changes.

---

## Feature Pairs Studied

Three environmental feature pairs are investigated:

1. **Temperature + Humidity**
2. **Temperature + Windspeed**
3. **Humidity + Windspeed**

For each pair, the following are measured:

- Effect of shifting feature A individually
- Effect of shifting feature B individually
- Effect of shifting both features simultaneously
- Expected additive effect
- Interaction effect

---

## Interaction Effect

The interaction effect is calculated as:

```text
Interaction =
Actual combined effect − Expected additive effect
```

where:

```text
Expected additive effect =
Effect of feature A + Effect of feature B
```

The effect of a shift is defined as:

```text
Shift effect = Shifted MAE − Baseline MAE
```

### Interpretation

- **Positive interaction** → combined shift causes more degradation than expected from adding the individual effects.
- **Negative interaction** → combined shift causes less degradation than expected from adding the individual effects.
- **Interaction close to zero** → combined effect is approximately additive.

---

## Main Results

The primary Linear Regression experiment with a `+0.10` shift produced the following interaction effects:

| Feature Pair | Interaction Effect |
|---|---:|
| Temperature + Humidity | **-3.782** |
| Temperature + Windspeed | **+0.189** |
| Humidity + Windspeed | **-0.179** |

### Main Observation

The **Temperature + Humidity** pair shows a clearly negative interaction effect:

```text
Interaction ≈ -3.782
```

This means that the combined MAE degradation was **smaller than the sum of the individual MAE effects**.

In contrast, the Temperature + Windspeed and Humidity + Windspeed interactions are relatively close to zero, suggesting that their combined effects are approximately additive under this experimental setup.

---

## Robustness Analysis

### Random Forest

A Random Forest Regressor was used as an additional robustness check.

For the Temperature + Humidity pair, the Random Forest produced a negative interaction effect of approximately:

```text
-1.819
```

The direction of the interaction therefore remained negative, although its magnitude differed from the Linear Regression result.

This suggests that the qualitative pattern is not limited entirely to the Linear Regression model.

---

## Random Seed Robustness

The Temperature + Humidity experiment was repeated using different train/test split random seeds.

The interaction effects were approximately:

| Random Seed | Interaction |
|---:|---:|
| 42 | -3.782 |
| 21 | -3.870 |
| 7 | -4.051 |

The interaction remained negative across all tested seeds.

For the other feature pairs, the interaction effects remained relatively close to zero across the tested seeds.

---

## Boundary-Safe Sensitivity Analysis

The primary shift uses:

```python
(x + 0.10).clip(0, 1)
```

Because some feature values are already close to the upper boundary, clipping can affect a small portion of the observations.

To examine whether this influenced the main finding, an alternative boundary-safe shift was tested:

```python
x_shifted = x + 0.10 * (1 - x)
```

This moves each value 10% closer to 1 without exceeding the upper boundary.

For Temperature + Humidity:

```text
Original +0.10 interaction:
-3.782

Boundary-safe interaction:
-0.828
```

The magnitude changed, but the interaction remained negative.

This supports the conclusion that the **direction of the Temperature + Humidity interaction is reasonably robust to the choice of shift implementation**, although its exact magnitude depends on the shift definition.

---

## Additional Analysis

The Linear Regression model produces additive changes in predictions when the individual feature shifts are combined.

However, the changes in MAE are not necessarily additive.

This occurs because MAE is based on the absolute value of individual prediction errors:

```text
MAE = mean(|actual − prediction|)
```

Even when prediction changes are additive, the absolute-value operation can cause the resulting MAE changes to behave non-additively.

Therefore, the observed interaction in MAE should not automatically be interpreted as proof of a nonlinear interaction between the underlying features.

Instead, the result demonstrates that **model-performance degradation under simultaneous shifts can differ from the simple sum of individual performance effects**.

---

## Key Findings

The study provides several main observations:

1. Simultaneous feature shifts do not always produce purely additive changes in model performance.
2. The Temperature + Humidity pair produced the clearest non-additive effect in the main experiment.
3. Temperature + Windspeed was approximately additive.
4. Humidity + Windspeed was also approximately additive.
5. The negative direction of the Temperature + Humidity interaction remained across multiple random seeds.
6. A Random Forest model also produced a negative interaction for Temperature + Humidity.
7. The boundary-safe sensitivity analysis preserved the negative direction, although the magnitude changed.
8. The results are specific to the dataset, model, shift magnitude, and experimental design used in this study.

---

## Limitations

This is an exploratory study, and the findings should not be interpreted as universal properties of the features or dataset.

Important limitations include:

- Only one primary shift magnitude (`+0.10`) was used in the main experiments.
- An alternative boundary-safe shift was used as a sensitivity check.
- Only three environmental feature pairs were investigated.
- The primary analysis uses Linear Regression.
- The Random Forest analysis is used as a robustness check rather than an exhaustive model comparison.
- The study uses a single dataset.
- The train/test split is random rather than explicitly time-based.
- The shift is artificially constructed and does not necessarily represent a real-world distribution shift.
- The magnitude of the interaction can depend on the chosen performance metric and shift definition.
- The experiments do not establish causality between environmental variables and model degradation.

The primary experiments used a `+0.10` shift, with an alternative boundary-safe shift used as a sensitivity check.

---

## Conclusion

This exploratory study examined whether simultaneous shifts in multiple features cause model-performance degradation that is simply additive.

The results suggest that **non-additive effects can occur**.

The clearest example was the **Temperature + Humidity** pair, which produced a negative interaction effect under the primary Linear Regression experiment. This indicates that the combined degradation in MAE was smaller than would be expected from simply adding the two individual degradation effects.

The other feature pairs showed interaction effects close to zero, suggesting approximately additive behavior under the tested conditions.

The robustness and sensitivity analyses showed that the negative direction for Temperature + Humidity persisted across different random seeds, under a Random Forest model, and under an alternative boundary-safe shift.

However, the exact magnitude of the interaction changed across experimental conditions. Therefore, the findings should be interpreted as **evidence that simultaneous distribution shifts can produce non-additive model-performance effects in this experimental setting**, rather than as a universal rule.

---

## Future Work

Several extensions could make the study more comprehensive:

- Test multiple shift magnitudes such as `+0.05`, `+0.10`, `+0.20`, and `+0.30`.
- Investigate negative as well as positive shifts.
- Test additional feature combinations.
- Use time-based train/test splits.
- Compare additional machine-learning models.
- Examine interaction effects using other performance metrics.
- Investigate realistic distribution-shift scenarios rather than synthetic shifts.
- Study whether explicit feature interactions in the model change the observed behavior.
- Evaluate the experiments on additional datasets.

---

## Reproducibility

To reproduce the analysis:

1. Obtain the UCI Bike Sharing Dataset.
2. Use the `hour.csv` file.
3. Place the dataset in the appropriate location expected by the notebook.
4. Install the required Python libraries.
5. Open:

```text
notebooks/distribution_shift_analysis.ipynb
```

6. Run all cells from the beginning.

The notebook contains the complete analysis pipeline, including:

- Data exploration
- Feature selection
- Model training
- Baseline evaluation
- Distribution-shift experiments
- Interaction calculations
- Robustness checks
- Sensitivity analysis
- Visualizations
- Final interpretation

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Dataset Source

The dataset used in this project is:

**Bike Sharing Dataset — UCI Machine Learning Repository**

The dataset was originally collected from the Capital Bikeshare system and contains hourly and daily bike rental information together with weather and seasonal information.

Dataset source:

https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

---

## Project Structure

```text
Feature-Interactions-Under-Distribution-Shift/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── distribution_shift_analysis.ipynb
│
├── figures/
│   └── ...
│
└── report/
    └── Feature_Interactions_Distribution_Shift_Final_Report.pdf
```

---

## Author

**Ankita Arora**

This project was conducted as an exploratory study of machine-learning robustness under simultaneous feature distribution shifts.
