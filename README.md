### Author

* **Name:** Wei Liu
* **Student ID:** 1161456

### Project Introduction

Based on preliminary data exploration (EDA) from `EDA(updated).ipynb` and `Personal_health_lifesyle.ipynb`, the chosen dataset contains a clear target column with two distinct values: 'healthy' and 'diseased'. This characteristic makes the dataset an ideal case for a supervised binary classification task.

This project will therefore focus on applying the machine learning concepts and techniques covered in this course to address a clear objective: **predicting whether an individual is 'diseased' or 'healthy' based on their lifestyle features.**

To align with real-world medical priorities, **'diseased' will be treated as the positive class during training and evaluation**. The rationale is that the cost of a "False Negative" (failing to detect a 'diseased' patient) is significantly higher than the cost of a "False Positive" (incorrectly flagging a 'healthy' patient for a follow-up). This consideration will be central to the evaluation strategy.

### Project Workflow

1. **Data Processing (`data_processing.ipynb`):**

   * I started with the raw dataset, `health_lifestyle_classification.csv`.
   * In this notebook, I performed data cleaning, which included handling duplicate and irrelevant columns, dropping columns with excessive missing values, and correcting obvious data entry errors.
   * Ultimately, I saved the processed, clean dataset to `data/cleaned_data.csv`.
2. **Exploratory Data Analysis & Feature Selection (`EDA(updated).ipynb`):**

   * Using `cleaned_data.csv`, I conducted an in-depth Exploratory Data Analysis (EDA), focusing on personal lifestyle-related features.
   * The analysis revealed that no single feature has a strong correlation with health status, indicating that the relationships are complex and non-linear.
   * Based on these findings, I removed redundant features (e.g., replacing `weight` and `height` with `bmi_scaled`) and prepared the final dataset for modeling, `data/EDA_data.csv`.
3. **Modeling (`Assignment3.ipynb`):**

   * This is the core part of the project where I loaded `data/EDA_data.csv`.
   * **Feature Engineering:** To better capture the non-linear relationships, I constructed new composite features (e.g., `stress_workload`, `age_band`).
   * **Algorithm Selection:**
     * **Baseline Model:** I chose `LogisticRegression` as a baseline to test the hypothesis from the EDA—that a simple linear model would likely underfit and perform poorly.
     * **Primary Model:** I selected `HistGradientBoostingClassifier`. As a tree-based ensemble model, it is well-suited for handling the non-linear relationships and high-order feature interactions present in the data.
     * **Models Not Chosen:** I did not use `KNN` because the high-dimensional, sparse feature space after one-hot encoding makes distance metrics unreliable (the "curse of dimensionality"). **Unsupervised learning** algorithms were also not used, as the project has a clear, labeled target ('healthy'/'diseased'), making it a supervised classification problem.
   * **Evaluation & Optimization:**
     * I defined **Recall** as the primary evaluation metric to minimize the number of missed 'diseased' patients.
     * After using `GridSearchCV` for hyperparameter tuning, I performed **threshold tuning** and successfully increased the Recall for the 'diseased' class to 100%, achieving the project's primary medical objective.
   * **Explainable AI (XAI):** Finally, I used SHAP to interpret the "black box" `HistGradientBoostingClassifier` model, identifying the key drivers behind its predictions.
