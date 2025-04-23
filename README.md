# Big Data Analytics Assignment 2

This Jupyter Notebook demonstrates three different Big Data analytics tasks using Apache Spark: Classification, Clustering, and Recommendation Engine.

---

## 1. Classification Model

This section builds a classification model using Spark's Logistic Regression to classify Iris flowers based on their features.

**Libraries Used:**

-   `pyspark.sql`: For Spark DataFrame operations.
-   `pyspark.ml.feature`: For feature engineering (StringIndexer, VectorAssembler).
-   `pyspark.ml.classification`: For Logistic Regression.
-   `pyspark.ml.evaluation`: For model evaluation.
-   `urllib.request`: For downloading the dataset.

**Workflow:**

1.  **Data Loading:** The Iris dataset is downloaded and loaded into a Spark DataFrame.
2.  **Feature Engineering:**
    -   StringIndexer converts the categorical 'species' column into numerical labels.
    -   VectorAssembler combines the feature columns into a single 'features' vector.
3.  **Model Training:**
    -   The data is split into training and testing sets.
    -   A Logistic Regression model is trained on the training data.
4.  **Prediction and Evaluation:**
    -   The model is used to make predictions on the test data.
    -   MulticlassClassificationEvaluator calculates the accuracy of the model.

**Key Results:**

-   Accuracy of the Logistic Regression model on the test data. (In this case, 1.00)

---

## 2. Clustering Model

This section builds a clustering model using Spark's K-Means algorithm to cluster Iris flowers.

**Libraries Used:**

-   `pyspark.sql`: For Spark DataFrame operations.
-   `pyspark.ml.feature`: For feature engineering (VectorAssembler).
-   `pyspark.ml.clustering`: For K-Means clustering.
-   `pyspark.ml.evaluation`: For clustering evaluation.
-   `urllib.request`: For downloading the dataset.

**Workflow:**

1.  **Data Loading:** The Iris dataset is downloaded and loaded into a Spark DataFrame.
2.  **Feature Engineering:** VectorAssembler combines the feature columns into a single 'features' vector.
3.  **Model Training:**
    -   A K-Means model is initialized with k=3 (as there are 3 species of Iris) and trained on the data.
4.  **Prediction and Evaluation:**
    -   The model is used to predict clusters for the data points.
    -   ClusteringEvaluator calculates the Silhouette Score to evaluate the clustering quality.

**Key Results:**

-   Silhouette Score indicating the quality of the clusters. (In this case, 0.74)
-   Cluster centers.

---

## 3. Recommendation Engine

This section builds a movie recommendation engine using Spark's ALS (Alternating Least Squares) algorithm.

**Libraries Used:**

-   `pyspark.sql`: For Spark DataFrame operations.
-   `pyspark.ml.evaluation`: For model evaluation (RegressionEvaluator).
-   `pyspark.ml.recommendation`: For the ALS algorithm.
-   `urllib.request`: For downloading the dataset.
-   `zipfile`: For unzipping the dataset.
-   `os`: For operating system utilities.

**Workflow:**

1.  **Data Loading:** The MovieLens 100K dataset is downloaded, extracted, and the ratings data is loaded into a Spark DataFrame.
2.  **Model Training:**
    -   The data is split into training and testing sets.
    -   An ALS model is trained on the training data to predict movie ratings.
3.  **Prediction and Evaluation:**
    -   The model is used to predict ratings on the test data.
    -   RegressionEvaluator calculates the Root Mean Square Error (RMSE) to evaluate the prediction accuracy.
4.  **Recommendation Generation:**
    -   The model is used to generate movie recommendations for all users.

**Key Results:**

-   Root Mean Square Error (RMSE) indicating the accuracy of the rating predictions. (In this case, 0.92)
-   Top 5 movie recommendations for each user.
