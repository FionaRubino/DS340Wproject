# DS340Wproject
## Dataset
The dataset file is included within this repository. The data is loaded through the path of the repository, so just run the cell that loads the data to use it for this code. The data was obtained from : . This data was already preprocessed and converted into numerical format for model preparation. 
 
## RUNNING THE NOTEBOOK
1. Clone the repository:
  In your terminal, run ```git clone https://github.com/FionaRubino/DS340Wproject.git```
2. Install Python if necessary. Verify you have Python 3.8+ installed with:
     ```python --version```
   If you do not have python, you can install it here: https://www.python.org/downloads/ .
3. Install Jupyter by running:
     ```pip install notebook```
4. Install required packages:
     ```pip install -r requirements.txt```
5. Start Jupyter Notebook:
    ```jupyter notebook```
   This will open a brower window.
6. Navigate to the notebook file: heartprediction.ipynb
7. Run all of the cells. Locate the Jupyter menu -> Kernel -> Restart & Run All. This runs all cells from start to finish.

## Results
Note that the results you will get in this notebook are different than those we reported in the paper and presentation. While no changes were made to the code, my final run of the results proposed different results, which may change explanations in the report and some concluding factors.  The results we recieved during my latest run are shown in the following tables. Below each table, we explain the differences between our original report.

### Logistic Regression

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy |  80.52 | 80.52 |  77.27  |
| Precision |  76.34 | 76.34  | 76.19 |
| Recall |  89.87  |  89.87 |  81.01  |
| F1 |  82.56 | 82.56 |  78.53  |

Almost all metrics remain the same as originally reported. There are slight changes in the final stage of our implementation with PCHF. Accuracy originally was also 77.27, but precision was 75.58, recall was 82.28, and F1 was 78.79. The differences are slight enough that our conclusions for LR are not impacted.

### SVM

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
|  Accuracy |  85.06  | 85.06 | 90.91 |
|  Precision  |  80.43 |  80.43 | 90.12 |
|  Recall |  93.67 | 93.67 | 92.41 |
|  F1  | 86.55 |  86.55 | 91.25 |

Baseline metrics remain the same. GridSearch-only metrics experienced some shifts. Originally, we reported 85.71% accuracy, 90.14% precision, 81.01% recall, and 85.33% F1. With PCHF, we originally reported 87.66% accuracy, 88.46% precision, 87.34% recall, and 87.9% F1. In the most recent run of code,  Grid, we see larger improvements from phase 2 to 3 than we did initally, but the improvement between phases still aligns with our conclusion that PCHF can improve this model.

### KNN

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
|  Accuracy | 75.97 | 96.75 | 98.05 |
|  Precision | 76.25 | 100 | 100 |
|  Recall | 77.22 | 93.67 | 96.2 |
|  F1 | 76.73  | 96.73 | 98.06 |

Baseline metrics remain the same. Phase 2 and Phase 3 results did change, but we still saw a major improvement after adding GridSearch, and a smaller improvement after adding PCHF. Therefore, our conclusions for KNN remain the same as originally reported.  Phase 2 initial results are: 99.35% accuracy, 100% precision, 98.73% recall, and 99.36% F1 Score. Phase 3 initial results are 100% across all performance metrics.

### XGBoost

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy | 98.05 | 96.75 | 98.05 |
| Precision | 100 | 97.44 | 100 |
| Recall | 96.2 | 96.2 | 96.2 |
| F1 | 98.08 | 96.82 | 98.06 |

Baseline results and GridSearch with PCHF remain the same. However, originally we observed 100% evaluation metrics across the board. This does change our interpretation of our results. In our report, we explained PCHF does not improve XGBoost, though results remained high. In the results from the most recent code run, we would say that it does improve model performance, and that tree-based models generally are improved by the PCHF process.

### Random Forest

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy | 96.75 |  96.75 | 98.05 |
| Precision |  100 |  100 | 100 |
| Recall |  93.67  | 93.67 | 96.2 |
| F1 | 96.73 | 96.73 | 98.06 |

Baseline results remain the same. Initially we found 100% evaluation metrics for all of phase 2 and phase 3. This changed in our recent run through. Now, GridSearch-only kept results the same as the baseline, and adding PCHF improved metrics, achieving a 98.05% accuracy, 100% precision, 96.2% recall, and 98.06% F1 Score. The slight improvement still indicated PCHF can be beneficial to this model and achieve high peformance.

### Decision Tree

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy |  98.05 |  94.81 | 98.05 |
| Precision |  100 | 96.1 | 100  |
| Recall |  96.2   | 93.67 | 96.2 |
| F1 |  98.08 | 94.87 | 98.06 |

Baseline and GridSearch-only results remained the same. Adding PCHF improved the results as expected. Originally, we reported 100% performance metrics across the board. Though the perfect score was not achieve, we did still see a significant improvement, so our conclusion about Decision Tree remains the same as dicussed in the paper.

