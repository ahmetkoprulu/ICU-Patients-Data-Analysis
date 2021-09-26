# Objective
This project includes analysis of ICU Patients Data which contains the clinical and demographic information of intensive care unit patients. 
The issue that the project worked on is predicting outcome of the discharge of the patient. Why is this issue important is because discharging
patients quicker has the potential to increase the frequency of ICU readmissions but also quick discharges
can lead to death on the other hand slow discharges may not be effective in terms of cost. Since we are
predicting the outcome of discharge it is going to be a classification problem and the dataset we have
contains the labels “death” and “readmission” so it a supervised classification problem.

# Constraints
- **Heterogeneity and Imbalance of Data**: 
Many algorithms like neural networks and support vector machines like their
feature vectors to be homogeneous numeric and normalized. The algorithms that
employ distance metrics are very sensitive to this, and hence if the data is
heterogeneous, these methods should be the afterthought. Decision Trees can handle
heterogeneous data very easily. Moreover, reflects an unequal distribution of classes within a
dataset. Most machine learning algorithms work best when the number of samples in each class
are about equal. This is because most algorithms are designed to maximize accuracy and reduce
error. Therefore, before training our models we standardize our numeric
variables in order to scale for the specified models. Also, applied one hot encoding to the
categorical variables since most of the algorithms we use works with numerical data.
Furthermore, we used ADASYN(Adaptive Synthetic Sampling Method) to balance mostly
negative death and readmitted targets.
- **Redundancy of Data/ Curse of Dimensionality**: 
If the data contains redundant information, i.e. contain highly correlated values,
then it’s useless to use distance based methods because of numerical instability. In
this case, some sort of Regularization can be employed to the data to prevent this
Situation. Moreover, if the data scientist can
manually remove irrelevant features from the input data, this is likely to improve the
accuracy of the learned function. In addition, there are many algorithms for feature
selection that seek to identify the relevant features and discard the irrelevant ones. Therefore, to
compute relations between features and targets we mainly compare metrics such as pearson
correlation score and p-values. Also, Lasso (L2) regularization will down effectiveness of
unrelated features to zero and eliminate it.
- **Dependent Features**: 
If there is some dependence between the feature vectors, then algorithms that
monitor complex interactions like Neural Networks and Decision Trees fare better
than other algorithms.
# Dataset
In ICU Patients dataset there are 51 features and we will focus on death and readmitted variables.
Top 5 correlated features for death is like:
| Feature | Correlation |
|-----|----------|
| Age | 0.177991 |
| LOS | 0.073823 |
| previous_LOS | 0.058586 |
| previous_ICU_stays | -0.007155 |
| Charlson_index | 0.002592 |

Statistics about top 5 correlated features of death is like:

|Feature| Coef | Std Err| t | P > t | [0.025 0.975]|
|-------|-----|--------|--|------|--------------|
|Age| 0.0019| 0.000| 12.828| 0.000| 0.002| 0.002|
|LOS| -2.893e-06| 1.27e-05| -0.228| 0.819| -2.77e-05| 2.2e-05|
|previous_LOS| 0.0002| 3.02e-05| 6.037| 0.000| 0.000| 0.000|
|previous_ICU_stays| -0.0081| 0.005| -1.588| 0.112| -0.018| 0.002|
|Charlson_index| 0.0026| 0.008| 0.327| 0.744| -0.013| 0.018|

Top 5 correlated features for readmitted like:

| Feature | Correlation |
|-----|----------|
| Age | 0.041844 |
| LOS | -0.011790 |
| previous_LOS | 0.062318 |
| previous_ICU_stays | 0.040352 |
| Charlson_index | 0.005481 |

Statistics about top 5 correlated features of readmitted is like:

|Feature| Coef | Std Err| t | P > t | [0.025 0.975]|
|-------|-----|--------|--|------|--------------|
|Age| 0.0011| 0.000| 5.903| 0.000| 0.001| 0.001|
|LOS| -8.48e-06| 1.58e-05| -0.537| 0.591| -3.94e-05| 2.25e-05|
|previous_LOS| 0.0001| 3.76e-05| 3.902| 0.000| 7.31e-05| 0.000|
|previous_ICU_stays| 0.0193| 0.006| 3.021| 0.003| 0.007| 0.032|
|Charlson_index| -0.0056| 0.010| -0.564| 0.573| -0.025| 0.014|

By looking at the statistics above, even though a feature is good enough correlated it still may not be
statistically significant. Therefore, one of them may work for the model and one of them may not. Thats
why rather than computing relations between variables manually, we used recursive feature elimination
methods to achieve that.

Also, the dataset is unbalanced(overwhelmingly negative) we oversampled the normal dataset by using
ADASYN and create over sampled dataset. In total there are three datasets total like normal, oversampled
and feature eliminated.

# Results
For results please check the analysis report
