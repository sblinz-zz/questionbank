# Question Bank

## Project Description

A machine learning library for test question data. Analyze, recommend effective questions, and predict student responses.

## Design Notes

Designed for data with samples containing student id, question id, score, and (optionally) maximum possible scores for each question.

### Ingest

*Reshape data to view questions as features contributing to student grades.
*Add maximum possible score per question for weighted grading.

### Analysis

*Normalize individual scores by maximum possible score.
*Compute grades as average of individual scores or ratio of total score to maximum possible score on attempted questions.
*Plot histograms and desnity scatter plots of variables.

### Data Completion

With a question bank containing many questions, most students will only have attempted a select number. Since some of the more interesting recommendation methods require complete data, we consider data completion using means

*Column or row-based mean completion

### Recommendation

By viewing each question as a feature contributing to a students overall accuracy we can use various feature selection methods to eliminate ineffective questions which do not offer a statistically significant contribution to student grades. We consider

*Variance-based elimination (doesn't require data completion)
*Linear Regression
*Random Forest

### Prediction

Predict student scores on a given question using results from other students. Details TBD.