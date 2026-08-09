# Analytics Module - Titanic Survival Analysis

This module covers the full data analysis and machine learning pipeline
on the classic Titanic dataset. The goal is to understand what kind of
passengers were more likely to survive and then build models to predict
survival. The dataset is loaded once from seaborn and every step below
works from that same data.

The work is split into two notebooks.
01_eda.ipynb handles all the data exploration and cleaning.
02_modeling.ipynb handles all the machine learning work.
A saved file called titanic.csv is committed here as an offline backup
so grading works even without internet.

---

## Section 1 - Loading the Dataset and Profiling It

The very first thing I did was load the Titanic dataset using seaborn's
built-in loader. This gives us information about 891 passengers who were
on the Titanic in 1912.

Right after loading I saved the data to a file called titanic.csv using
df.to_csv so that anyone grading the work can reload it with pd.read_csv
even if the internet is not available.

To understand the data I printed three things.

df.info() showed me all the column names, their data types, and how many
non-null values each column has. There are 15 columns in total.

df.describe() gave me the statistical summary like mean, minimum, maximum,
and standard deviation for all the numeric columns.

df.shape told me the dataset has 891 rows and 15 columns.

I also sorted the fare column from highest to lowest to see how spread out
the ticket prices were. Some passengers paid over 500 pounds for their ticket.

---

## Section 2 - Handling Missing Values

Not all columns had complete data. Some passengers had missing information.
I measured the exact percentage of missing values in every column.

The columns that had missing values were:

age had 19.87 percent missing values.
embarked had 0.22 percent missing values.
embark_town had 0.22 percent missing values.
deck had 77.22 percent missing values.

**Approach (percentage-based threshold rule):** I used a simple threshold
rule to decide what to do with each one.

Under 5 percent missing means drop those rows.
Between 5 and 30 percent missing means impute or fill the values.
More than 30 percent missing means drop the column or treat missing as
its own category.

**Design Decision (embarked, 0.22%, < 5% threshold → drop rows):** For
embarked at 0.22 percent I dropped the 2 rows that had null values because
losing only 2 rows out of 891 is completely fine.

**Design Decision (embark_town, 0.22%, < 5% threshold → drop rows):** For
embark_town at 0.22 percent I did the same thing and dropped those rows.

**Design Decision (age, 19.87%, in the 5-30% band → median impute):** For
age at 19.87 percent I filled the missing values with the median age. The
median is a good choice here because age has some very old passengers that
would push the mean too high. The median is more stable and honest.

**Design Decision (deck, 77.22%, > 30% threshold → drop the column):** For
deck at 77.22 percent I decided to drop the entire column. If I tried to
fill 77 percent of the column I would be making up most of the data which
is not a good idea. Dropping those rows instead would lose almost 80 percent
of the whole dataset which is even worse. So dropping the column was the
only sensible choice.

---

## Section 3 - Univariate Analysis

Univariate analysis means looking at one column at a time to understand
how its values are spread out.

I made a histogram and a box plot for both age and fare.

The age histogram showed most passengers were between 20 and 40 years old.
The age box plot showed the range and a few dots outside the whiskers which
are the outliers.

The fare histogram showed most passengers paid very cheap tickets and only
a small number paid very high amounts. This creates a long right tail.

The fare box plot also showed many dots far to the right which are passengers
who paid extremely high fares for first class cabins.

**Approach (IQR outlier rule):** To count the outliers I used the IQR rule.
An outlier is any value that falls below Q1 minus 1.5 times IQR or above
Q3 plus 1.5 times IQR.

For age the IQR was 13.0 with a lower bound of 2.5 and upper bound of 54.5.
After imputing the missing ages, age had 65 outliers.
Before imputation using the original saved CSV, age had only 11 outliers.
Filling missing ages with the median adds many values near the centre which
changes the IQR and makes more values appear to be outliers.

For fare the IQR was 23.10 with a lower bound of minus 26.76 and upper bound
of 65.66. Fare had 114 outliers. These are mainly first class passengers who
paid very high amounts.

I also computed the mean, median, and mode for the fare column.

Mean was 32.10
Median was 14.45
Mode was 8.05

Since mean (32.10) is greater than median (14.45) which is greater than mode
(8.05) the fare distribution is right-skewed or positively skewed. This
happens because a small group of wealthy passengers paid very high fares which
pulls the mean far above where most passengers actually paid.

---

## Section 4 - Bivariate Analysis

Bivariate analysis means comparing two columns at a time to find patterns.

I used boolean masking with the ampersand operator to filter the data and
compute survival rates.

Part (a) - Survival rate by sex

Male passengers: 577 total, 109 survived. Male survival rate was 18.89 percent.
Female passengers: 312 total, 231 survived. Female survival rate was 74.04 percent.
Women survived at almost four times the rate of men. This strongly suggests
the evacuation followed a women first approach.

Part (b) - Survival rate by passenger class

First Class: 62.62 percent survival rate.
Second Class: 47.28 percent survival rate.
Third Class: 24.24 percent survival rate.
Survival dropped sharply as class number went up. First class had access to
better cabin locations and more priority during evacuation.

Part (c) - Survival rate by sex and class together

Female First Class: very high survival rate close to 97 percent.
Female Second Class: also very high survival rate close to 92 percent.
Female Third Class: survival rate around 50 percent.
Male First Class: survival rate around 37 percent.
Male Second Class: survival rate around 16 percent.
Male Third Class: survival rate around 14 percent.
Even female passengers in third class survived at a higher rate than male
passengers in first class. This shows sex was even more important than class.

**Design Decision (exclude adult_male/alone from the correlation matrix):**
I then computed a correlation matrix on exactly six columns: survived, pclass,
age, sibsp, parch, and fare. I did not include adult_male or alone because
adult_male is directly computed from sex and age and alone is computed from
sibsp plus parch being zero. Including them would be repeating the same
information and they are not independent features.

The 6x6 correlation matrix was displayed as a heatmap using sns.heatmap.

The two strongest off-diagonal correlations by absolute value were:

1. pclass and fare with a coefficient of minus 0.55.
This is a strong negative relationship. First class tickets were very expensive
and third class tickets were cheap. As the class number goes from 1 to 3 the
fare drops a lot.

2. sibsp and parch with a coefficient of plus 0.41.
Passengers who traveled with siblings or spouses were also more likely to travel
with parents or children. Families tend to travel together as a whole unit.

---

## Section 5 - Multivariate Data Story

Multivariate analysis means looking at more than two columns together to build
a bigger picture of what was happening.

I made five charts to tell a connected story about who was more likely to survive.

Chart 1 - Survival Rate by Sex (Bar Chart)

This bar chart compared the survival rate for male and female passengers.
Females had a much higher survival rate than males. This was the single strongest
factor in predicting survival. The evacuation clearly gave priority to women and
children over men regardless of class.

Chart 2 - Survival Rate by Passenger Class (Bar Chart)

This bar chart showed that first class passengers survived the most and third class
passengers survived the least. The higher the class number the lower the survival rate.
First class cabins were on upper decks which gave those passengers faster access to the
lifeboats. Third class passengers had to travel further and faced more barriers.

Chart 3 - Age Distribution by Survival (Box Plot)

This box plot compared the age distribution of survivors and non-survivors.
Both groups had a similar spread of ages so age alone was not a strong predictor.
However the median age of survivors was slightly lower. Young children were given
some priority during evacuation which contributed to this small difference.

Chart 4 - Age vs Fare Colored by Survival (Scatter Plot)

This scatter plot put age on the x axis and fare on the y axis and colored each
dot by whether the passenger survived or not. Passengers in the upper part of
the chart who paid high fares were almost all survivors. The dense cluster of
passengers who did not survive was in the low fare region across all age groups.
This shows that fare, which is tied to class, was a key factor in survival.

Chart 5 - Survival Rate by Sex and Class Combined (Bar Chart with hue)

This grouped bar chart showed survival rates for male and female passengers broken
down by class. Female passengers had higher survival rates than males in every class.
First class females survived at the highest rate. Even females in third class survived
more than males in first class. This chart shows that sex and class worked together
and combining them gives the clearest picture of survival chances.

Final Conclusion from the Data Story

Survival on the Titanic was mainly driven by sex and passenger class. Women were
strongly prioritized during the evacuation. First class passengers had physical and
social advantages that gave them better access to lifeboats. Fare reflects class so
it also correlates with survival. Age had a smaller role but children were given some
priority. Together sex and class explain most of what we see in the survival data.

---

## Section 6 - Z-Score Standardization as an EDA Check

**Design Decision (EDA-only check, not reused by modeling):** Before doing
any machine learning it is a good idea to check how standardization would
affect the numeric features. This is just an exploratory check and does not
feed into the modeling pipeline. The modeling section does its own
train-only scaling with a fresh StandardScaler fit inside the Pipeline, so
this EDA-stage scaler is never reused for anything downstream.

I standardized age and fare using the z-score formula which is z equals x minus
mean divided by standard deviation. I used StandardScaler from scikit-learn on a
copy of the cleaned data so the original was not changed.

Before standardization the stats were:
age had a mean of 29.32 and a standard deviation of 12.99.
fare had a mean of 32.10 and a standard deviation of 49.70.

After standardization the stats were:
age had a mean of 0.000 and a standard deviation of 1.001.
fare had a mean of 0.000 and a standard deviation of 1.001.

Both columns now have approximately mean 0 and standard deviation 1 which confirms
the scaling worked correctly. The shape of the distribution stays the same.
Only the scale changes.

---

## Section 7 - Train and Test Split with Stratification

Before doing any machine learning the data needs to be split into a training set
and a testing set. The model learns from the training set and is evaluated on the
testing set which it has never seen before.

**Design Decision (stratified split before any preprocessing):** I used a
stratified split with an 80 percent training and 20 percent testing ratio.

Stratification is important here because the survived column is not balanced.
About 61.6 percent of passengers did not survive and only 38.4 percent survived.
Without stratification a random split might accidentally put most survivors in the
training set and very few in the testing set. That would make the evaluation results
unreliable because the model never gets tested on a fair sample.

Stratify makes sure both the training set and the testing set have the same class
proportions as the original dataset.

After splitting the distributions were:
Original: 61.6 percent not survived, 38.4 percent survived.
Training: 61.7 percent not survived, 38.3 percent survived.
Testing: 61.5 percent not survived, 38.5 percent survived.
The proportions match very closely which confirms stratification worked correctly.

---

## Section 8 - Preprocessing Pipeline

Preprocessing means getting the data ready for the model. This includes filling
missing values, converting text columns to numbers, and scaling numeric values.

**Design Decision (fit-on-train-only, no data leakage):** A very important
rule is that every preprocessing step must be fit only on the training data.
The test data should only be transformed using what was learned from the
training data. Fitting on the test data or on the full dataset before
splitting would leak test information into training which makes results
misleading.

**Approach (Pipeline + ColumnTransformer enforces the rule structurally):**
I used scikit-learn's Pipeline and ColumnTransformer to handle this
automatically instead of relying on remembering to call fit vs. transform
correctly by hand.

The features I selected were pclass, sex, age, sibsp, parch, fare, and embarked.
The target was survived.

For numeric features (age, fare, sibsp, parch, pclass) I used:
SimpleImputer with median strategy to fill any remaining missing values.
StandardScaler to scale the values so they have mean 0 and std 1.

For categorical features (sex, embarked) I used:
SimpleImputer with most frequent strategy to fill any remaining missing values.
OneHotEncoder to convert text categories like male and female into numeric columns.

All of this was wrapped in a ColumnTransformer which applies the right pipeline
to each group of columns automatically. The whole thing was placed inside a Pipeline
so the fit and transform steps are always kept separate and correct.

---

## Section 9 - Training Three Classifiers

I trained three different classification models on the same training data and tested
all of them on the same test data so the comparison is fair.

Model 1 - Logistic Regression

Logistic Regression is a simple model that draws a straight line to separate survived
from not survived. It works well as a baseline and is easy to interpret.
I put the preprocessor and LogisticRegression together in a Pipeline and fit it on
the training data.

Model 2 - Decision Tree

A Decision Tree asks a series of yes or no questions to classify each passenger.
For example it might ask if the passenger is female and then ask what class they were in.
I also visualized the tree using plot_tree from scikit-learn with feature names and class
names labeled so the splits are easy to read.

Model 3 - Random Forest

A Random Forest builds many decision trees and combines their votes. This reduces the
chance of overfitting that a single decision tree can have. I used 100 trees in the forest.

All three models were built using the same preprocessor so they all saw identical input data.
This guarantees the comparison between them is meaningful.

---

## Section 10 - Evaluating All Three Models

To understand how well each model performs I measured several metrics on the test set.

Confusion Matrix shows how many predictions were correct and how many were wrong.
It splits results into true positives, true negatives, false positives, and false negatives.

Accuracy is the percentage of all predictions that were correct.
Precision is how many of the predicted survivors were actually survivors.
Recall is how many of the actual survivors the model correctly found.
F1 Score is a balance between precision and recall.
AUC is the area under the ROC curve and shows how well the model separates the two classes.

Results:

Logistic Regression scored Accuracy 0.804, Precision 0.793, Recall 0.667,
F1 Score 0.724, and AUC 0.844.

Decision Tree scored Accuracy 0.816, Precision 0.773, Recall 0.739,
F1 Score 0.756, and AUC 0.797.

Random Forest scored Accuracy 0.816, Precision 0.800, Recall 0.696,
F1 Score 0.744, and AUC 0.827.

I also plotted an ROC curve for all three models on the same chart. The ROC curve
shows how the true positive rate changes as the false positive rate changes.
Logistic Regression had the highest AUC of 0.844 which means it is the best at
ranking survivors correctly even though its accuracy is slightly lower.

---

## Section 11 - Imbalance Handling Comparison

The survived column is not balanced. About 62 percent of passengers did not survive
and about 38 percent survived. This imbalance can cause models to favor the majority
class and miss many of the actual survivors.

I compared three strategies to handle this using Logistic Regression.

Strategy 1 - Baseline with no imbalance handling.
The model was trained as usual with no adjustments.
Precision 0.793, Recall 0.667, F1 Score 0.724.

Strategy 2 - class_weight balanced.
The model was told to pay more attention to the minority class by setting
class_weight to balanced. This makes the model treat each class more equally.
Precision 0.730, Recall 0.783, F1 Score 0.755.

Strategy 3 - SMOTE oversampling on training data only.
**Design Decision (SMOTE fit on the training fold only):** SMOTE creates new
synthetic examples of the minority class to make the training data more
balanced. I applied SMOTE only to the training fold to avoid leaking test
data information into the training process.
Precision 0.740, Recall 0.783, F1 Score 0.761.

Conclusion:

The baseline model had the highest precision but the lowest recall. It missed many
survivors because it was biased toward predicting not survived.
Both class_weight balanced and SMOTE improved recall significantly from 0.667 to 0.783.
SMOTE gave the best F1 Score at 0.761 which is a good balance between precision and recall.
If the goal is to catch as many survivors as possible then SMOTE or class_weight balanced
is a better choice than the baseline.

---

## Section 12 - Hyperparameter Tuning with GridSearchCV

Hyperparameter tuning means finding the best settings for a model by trying different
combinations and seeing which one performs the best.

I used GridSearchCV to search over three parameters of the Random Forest.

n_estimators controls how many trees are in the forest. I tried 50, 100, and 200.
max_depth controls how deep each tree can grow. I tried None (unlimited), 5, 10, and 20.
max_features controls how many features each tree considers at each split.
I tried sqrt, log2, and None.

GridSearchCV tested every combination of these settings using 5-fold cross validation.
That means it split the training data into 5 parts and trained and tested on each part.

The best parameters found were:
max_depth: 5
max_features: None
n_estimators: 50

The best cross validation score was 0.8189 which means the model correctly predicted
about 81.9 percent of cases on average during cross validation.

The OOB score (out of bag score) was 0.8076. The OOB score is computed using the
samples that were not included in each tree's training. It is an honest estimate of
how well the model generalizes to new data without needing a separate validation set.
**Requirement (oob_score=True must be set at construction time):** I set
oob_score=True when creating the RandomForestClassifier so this score was
available — passing it afterward would not work, oob_score_ is only
populated when the flag is on before fitting.

---

## Section 13 - Regression Side Task

As an extra task I built a multivariate linear regression model to predict fare
from the other available features. This is different from the classification task
where we predicted survived.

The features I used to predict fare were pclass, sex, age, sibsp, parch, and embarked.
I used a separate 80-20 train-test split for this task so it does not interfere with
the classification pipeline.

The preprocessing pipeline for regression was the same structure as before.
Numeric features were imputed with median and scaled.
Categorical features were imputed with most frequent and one-hot encoded.

The model was trained using LinearRegression from scikit-learn.

The evaluation metrics were:

MAE (Mean Absolute Error): 20.809
This means on average the model's fare predictions were about 20.81 pounds off
from the actual fare.

RMSE (Root Mean Squared Error): 30.473
This penalizes large errors more than MAE. The higher RMSE compared to MAE tells
us there are some large prediction errors for expensive tickets.

R2 (R-squared): 0.400
This means the model explains about 40 percent of the variation in fare. That is
a moderate fit. There is still a lot of variation that the model cannot capture.

Adjusted R2: 0.368
This adjusts R2 for the number of features used. It is always a bit lower than R2
and gives a more honest picture of model quality.

The residual plot showed that the spread of residuals grew larger as the predicted
fare went up. This fan shaped pattern is called heteroscedasticity. It means the
model is less accurate for passengers who paid very high fares and more accurate
for passengers who paid low or average fares.

---

## Section 14 - Final Model Comparison and Recommendation

Here is a summary of all three classification models side by side.

Classification Model Results:

Logistic Regression: Accuracy 0.804, Precision 0.793, Recall 0.667, F1 0.724, AUC 0.844
Decision Tree: Accuracy 0.816, Precision 0.773, Recall 0.739, F1 0.756, AUC 0.797
Random Forest: Accuracy 0.816, Precision 0.800, Recall 0.696, F1 0.744, AUC 0.827

Regression Model Results for fare prediction:

MAE: 20.809
RMSE: 30.473
R2: 0.400
Adjusted R2: 0.368

Note: The classification metrics and regression metrics are on completely different
scales and measure different things. They are listed separately and are not compared
to each other.

Final Recommendation:

**Design Decision (deployment choice):** Among the three classifiers I
would recommend the Random Forest for deployment.
Random Forest achieved the highest accuracy at 0.816 and a strong AUC of 0.827.
Its precision was the highest at 0.800 which means when it predicts a passenger
survived it is usually right. It also generalizes better than a single Decision Tree
because it averages the votes of many trees which reduces overfitting.
Logistic Regression had the highest AUC at 0.844 which is impressive for a simple
model and makes it a good backup choice when interpretability matters.
The Decision Tree had the best recall at 0.739 which means it catches more survivors
but at the cost of more false alarms.
For a balanced deployment where both accuracy and reliable positive predictions matter
Random Forest is the best overall choice based on the metric values observed.

---

## Section 15 - Saving and Reloading the Pipeline

The final step was to save the complete trained pipeline to disk so it can be reused
without retraining.

Section 14 recommended Random Forest as the model to deploy, so that is the
model I actually saved here, using the tuned hyperparameters found by
GridSearchCV in Section 12 (max_depth 5, max_features None, n_estimators 50)
so the saved artifact matches the written recommendation instead of quietly
saving a different model. I saved this Random Forest pipeline using
joblib.dump. The saved file is called titanic_survival_pipeline.pkl and is
stored inside the analytics folder.

The saved artifact includes both the full preprocessing pipeline and the trained
classifier together as a single object. This is important because saving only the
classifier without the preprocessor would mean new data would need to be manually
cleaned before prediction which is error prone. The saved pipeline handles all of
that automatically.

To verify the saved pipeline works I reloaded it using joblib.load and tested it
on three raw passengers that had never been preprocessed.

The raw passengers were:
A 22 year old male in third class who paid 7.25 pounds and boarded at Southampton.
A 38 year old female in first class who paid 71.28 pounds and boarded at Cherbourg.
A female in second class with no age recorded who paid 13.00 pounds.

The loaded pipeline predicted:
First passenger: did not survive (probability 0.07).
Second passenger: survived (probability 1.00).
Third passenger: survived with missing age handled automatically by the imputer
(probability 0.96).

The reloaded pipeline test accuracy was 0.7989 which is exactly the same as the
in-memory pipeline accuracy of 0.7989. The predictions on the test set were
identical for both confirming the save and reload worked correctly. This number
is the tuned Random Forest's accuracy specifically, so it is a little different
from the 0.816 accuracy reported for the untuned Random Forest back in Section 10,
since GridSearchCV optimized for cross-validation score rather than this exact
test split.

---

## Files in this Folder

01_eda.ipynb is the notebook for data loading, cleaning, and all EDA sections.
02_modeling.ipynb is the notebook for all machine learning tasks.
titanic.csv is the saved offline backup of the loaded dataset.
titanic_survival_pipeline.pkl is the saved trained pipeline ready for deployment.
charts/ holds a PNG export of every chart in both notebooks (all 10 EDA
charts from Section 3-5, the decision tree from Section 9, the ROC curve
from Section 10, and the residual plot from Section 13), saved automatically
via plt.savefig() right before each plt.show() call, so the charts are
available as standalone image files and not just embedded notebook outputs.
README.md is this file explaining every section of the project.
