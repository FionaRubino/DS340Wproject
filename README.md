# DS340Wproject
## Dataset
The dataset file is included within this repository. The data is loaded through the path of the repository, so just run the cell that loads the data to use it for this code. 
## Implementation
There is not much to running this notebook successfully. We have used a Jupyter Notebook to make it easy to interpret on your end. Simply, run each cell one after the other. Begin with loading the necessary imports. Then, run the baseline section to see the model results from the first part of the implementation. There will be 6 model blocks to run in the following order: Logistic Regression, SVM, KNN, XGBoost, Random Forest, and Decision Tree. Then, there is a pandas dataframe that was created to compare these results. Next, run the GridSearchCV section cell by cell, following the same order of models. There is no comparative table in this section. Finally, run the third and final section - combining GridSearchCV with PCHF feature engineering. First, run the cell containing the pchf block. Next, run the cells in same order as the previous two sections. The results from the third section are the results we use for the main analysis in the paper.
## Results
Note that the results you will get in this notebook are different than those we reported in the paper and presentation. While no changes were made to the code, my final run of the results proposed different results, which may change explanations in the report and some concluding factors.  The results we recieved during my latest run are shown in the following tables. Below each table, we explain the differences between our original report.

### Logistic Regression

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy |  80.52 | 80.52 |  77.27  |
| Precision |  76.34 | 76.34  | 76.19 |
| Recall |  89.87  |  89.87 |  81.01  |
| F1 |  82.56 | 82.56 |  78.53  |


### SVM

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
|  Accuracy |  85.06  | 85.06 | 90.91 |
|  Precision  |  80.43 |  80.43 | 90.12 |
|  Recall |  93.67 | 93.67 | 92.41 |
|  F1  | 86.55 |  86.55 | 91.25 |

### KNN

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
|  Accuracy | 75.97 | 96.75 | 98.05 |
|  Precision | 76.25 | 100 | 100 |
|  Recall | 77.22 | 93.67 | 96.2 |
|  F1 | 76.73  | 96.73 | 98.06 |

### XGBoost

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy | 98.05 | 96.75 | 98.05 |
| Precision | 100 | 97.44 | 100 |
| Recall | 96.2 | 96.2 | 96.2 |
| F1 | 98.08 | 96.82 | 98.06 |

### Random Forest

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy | 96.75 |  96.75 | 98.05 |
| Precision |  100 |  100 | 100 |
| Recall |  93.67  | 93.67 | 96.2 |
| F1 | 96.73 | 96.73 | 98.06 |

### Decision Tree

| Metric | Baseline | GridSearch | GridSearch & PCHF |
|-------|-----------|--------|-------------|
| Accuracy |  98.05 |  94.81 | 98.05 |
| Precision |  100 | 96.1 | 100  |
| Recall |  96.2   | 93.67 | 96.2 |
| F1 |  98.08 | 94.87 | 98.06 |
