# Detecting Heavy Drinking Episodes Using Smartphone Accelerometer Data

**Group Members:** Amy Griffiths, Juhee Lee
**Course:** MATH 4650 — Probabilistic Deep Learning
**Dataset:** [Bar Crawl: Detecting Heavy Drinking (UCI)](https://archive.ics.uci.edu/dataset/515/bar+crawl+detecting+heavy+drinking)

# Project Checkpoint

We have added the data. One CSV contained the accelerometer data while the Trandermal Alcohol Content (TAC) data was located in seperate files for each person. We first combined all of the seperate TAC CSV files into one data frame. We then merged that dataframe with the accelerometer data frame we created earlier based on the PID (Phone Identification). After we combined the data frames, we created a new "magnitude" column in order to reduce noise which is common in phone accelerometer data. We created "windows" of time of 2 seconds each so that our model is looking at seconds instead of milliseconds. We created features for these windows using mean/standard deviation/max/variance of the magnitudes as well as the standard deviation of the x, y, and z values. We then seperated our data into x's and y's and grouped them based on PID so what we could use Group K-Fold cross validation. If the model see's mostly phone A's data, it could recognize how that specific phone usually moves in the test set. We want our model to predict high TAC for any phone movement so we will be using random PID groups as the validation set. Finally, we created a pipeline and used cross validation. For this first model, we used a random forest. Our data is imbalanced though, so one of our next steps will be to use a weighted ramdom forest as well as possibly tying the SMOTE method. Because we want to use an ensemble method that takes many different models and votes on the label, this is just our first step towards that.

## Progress Update — Session 2

### What We Worked On

- Data loading and preprocessing pipeline for accelerometer and TAC data
- Timestamp alignment verification between accelerometer (milliseconds)
  and TAC (seconds → converted to milliseconds)
- Merge strategy: `merge_asof` with `direction='backward'`
  (assigns most recent prior TAC reading to each accelerometer row)
- Identified and resolved a critical bug: `tolerance=1800*1000`
  was incorrectly removing valid data — removed tolerance constraint
- Exploratory data analysis of TAC distributions across 13 participants

### What We Completed

- Accelerometer data loaded: 14,057,567 rows, 13 participants
- TAC data loaded: 715 rows (≈55 rows/participant at 30-min intervals)
- Timestamp unit mismatch identified and corrected (seconds → milliseconds)
- `merge_asof` pipeline validated per participant
- Binary label created: `drunk = 1 if TAC_Reading >= 0.08`
- Sliding window feature extraction implemented (window=100, step=50)
- More Features extracted: `mag_mean`, `mag_std`, `mag_max`, `mag_var`,
  `x_std`, `y_std`, `z_std`,+ `jerk_std`, +`sway`
- Trying 'Walk-forward validation framework' implemented to check seperating traning set better for group(person)
- Baseline ML models defined: Random Forest,
- 

### Key Findings

- SF3079 and JR8022 show ~99% drunk labels — confirmed as valid
  (accelerometer recording began after participant was already intoxicated)
- DK3500 shows 0% drunk labels — TAC never exceeded 0.08
- BK7610 and BU4707 share identical TAC readings — possible data issue
- Class imbalance: ~24% drunk vs ~76% sober across all participants

### What Is Next

* [ ]  create pipeline, again / seperating traning by drunk rate

* [ ] Gradient Boosting (with class_weight and SMOTE variants)

- [ ] Run walk-forward validation with all baseline ML models
- [ ] Evaluate using balanced accuracy
- [ ] Implement ResNet-18 (1D) for raw signal classification
- [ ] Compare ML feature-based vs ResNet end-to-end approaches
- [ ] Finalize participant split: train / validation / holdout test

---

## Repository Structure

```
├── 4650FinalProject.ipynb       # Main notebook
├── all_accelerometer_data_pids_13.csv
├── clean_tac/
│   ├── BK7610_clean_TAC.csv
│   ├── ...
│   └── SF3079_clean_TAC.csv
└── README.md
```

---

## References
