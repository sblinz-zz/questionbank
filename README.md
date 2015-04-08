# Question Bank

## Project Description

A machine learning library for analyzing test question data. Analyze, predict student responses, recommend effective questions

## Design Notes

Built around pandas DataFrames and designed for raw data with samples representing individual student responses on specific questions.

### Ingest

-Reshape data to view questions as features contributing to student grades.

-Add question difficulty ratings

-Add maximum possible score per question for weighted grading.

### Analysis

-Normalize individual scores by maximum possible score.

-Compute grades as average of individual scores or ratio of total score to maximum possible score on attempted questions.

-Plot histograms and desnity scatter plots of variables.

### Response Prediction and Data Completion

With a question bank containing many questions, most students will only have attempted a select number. Since some of the more interesting recommendation methods require complete data, we consider data completion using means

-Imputation (column or row mean-based completion)
-Distance Collaborative Filtering (supproting Pearson, Minkowski, and Cosine similarity metrics)
-Latent Collaborative Filtering (Matrix Factorization)

### Recommendation

By viewing each question as a feature contributing to a students overall accuracy we can use various feature selection methods to eliminate ineffective questions which do not offer a statistically significant contribution to student grades. We consider

-Variance-based elimination (doesn't require data completion)

-Linear Regression

-Random Forest