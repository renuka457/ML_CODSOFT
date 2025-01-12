# Movie Genre Classification

This project focuses on building a machine learning model that predicts the genre of a movie based on its plot summary or textual information. It utilizes techniques like TF-IDF for text vectorization and classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Workflow](#workflow)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Future Enhancements](#future-enhancements)

---

## Project Overview
Movies often belong to multiple genres that can be inferred from their plot summaries. This project aims to:
- Preprocess textual data (plot summaries).
- Vectorize the text using TF-IDF.
- Train and evaluate models to predict movie genres.

## Dataset
- **Source:** [IMDB Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- The dataset contains:
  - `movie_title`: Title of the movie.
  - `plot`: Plot summary of the movie.
  - `genre`: Genre(s) associated with the movie.

## Requirements
To run this project, you need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `nltk` (for text preprocessing)

Install the libraries using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk
