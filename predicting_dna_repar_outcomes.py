import numpy as np
import pandas as pd
import os
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb

data_dir = 'path_to_your_directory'
for dirname, _, filenames in os.walk(data_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data_dir = 'path_to_your_train.csv'
test_data_dir = 'path_to_your_test.csv'

df_train = pd.read_csv(train_data_dir)

df_train.head()

df_test = pd.read_csv(test_data_dir)

df_test.head()

seq_len = df_train['GuideSeq'].apply(len)

seq_len.describe()

# Check for missing values

df_train.isnull().sum()

# Describe the target variables

target_cols = ['Fraction_Insertions', 'Avg_Deletion_Length', 'Indel_Diversity', 'Fraction_Frameshifts']

df_train[target_cols].describe()

# Ensure X is a copy of the relevant part of df_train
X = df_train[['GuideSeq']].copy()

# Define one-hot encoding function
def one_hot_encode_sequence(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_sequence = [mapping[nuc] for nuc in sequence]
    return np.array(encoded_sequence).flatten()

# Apply one-hot encoding safely using .loc to avoid SettingWithCopyWarning
X.loc[:, 'GuideSeq_OHE'] = X['GuideSeq'].apply(one_hot_encode_sequence)

# Split 'GuideSeq_OHE' into individual columns
X_encoded = pd.DataFrame(X['GuideSeq_OHE'].tolist(), index=X.index)

# Split data
Targets = df_train.columns[2:]
X_train, X_val, Y_train, Y_val = train_test_split(X_encoded, df_train[Targets], test_size=0.3, random_state=42)

# Define base learners
#You can apply various other base learners (i.e.RandomForestRegressor, ExtraTreesRegressor, SVR, AdaBoostRegressor etc) to see which model performs best on your dataset.
base_learners = [
    ('GradientBoosting', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=2))]
# Train models and collect results
All_Results = []
for target in Targets:
    for name, model in base_learners:
        # Train model on training data
        model.fit(X_train, Y_train[target].values)
        # Collect predictions for validation data
        Predicted = model.predict(X_val)
        R2_Square = r2_score(Y_val[target].values, Predicted)
        All_Results.append({"Score": R2_Square, "Model": name, "Target": target})

# Convert results to DataFrame for better readability
results_df = pd.DataFrame(All_Results)
print(results_df)

from sklearn.metrics import r2_score

# Get the predictions for all targets
all_predictions = []
all_true_values = []

for target in Targets:
    for name, model in base_learners:
        # Train model on training data
        model.fit(X_train, Y_train[target].values)
        # Collect predictions for validation data
        Predicted = model.predict(X_val)

        all_predictions.append(Predicted)
        all_true_values.append(Y_val[target].values)

# Convert the lists to numpy arrays for r2_score computation
all_predictions = np.array(all_predictions).T  # Transpose to have targets as columns
all_true_values = np.array(all_true_values).T  # Transpose to have targets as columns

# Calculate multi-output R²
multi_output_r2 = r2_score(all_true_values, all_predictions, multioutput='uniform_average')
print(f"Multi-output R² for all targets: {multi_output_r2}")

# Perform one-hot encoding on 'GuideSeq' for the test set

X_test_encoded = pd.DataFrame(df_test['GuideSeq'].apply(one_hot_encode_sequence).tolist(), index=df_test.index)



# Initialize the submission DataFrame with the correct columns

submission = pd.DataFrame({

    'Id': df_test['Id'],

    'Fraction_Insertions': np.zeros(len(df_test)),

    'Avg_Deletion_Length': np.zeros(len(df_test)),

    'Indel_Diversity': np.zeros(len(df_test)),

    'Fraction_Frameshifts': np.zeros(len(df_test))

})



# Loop through each target, train the best model, and predict for the test set

for target in Targets:

    best_model_name, best_model = None, None

    best_r2_score = -np.inf



    # Find the best model for the current target

    for name, model in base_learners:

        model.fit(X_train, Y_train[target].values)

        predicted_val = model.predict(X_val)

        r2 = r2_score(Y_val[target].values, predicted_val)



        if r2 > best_r2_score:

            best_r2_score = r2

            best_model_name, best_model = name, model



    # Use the best model to make predictions on the test set for the current target

    submission[target] = best_model.predict(X_test_encoded)



# Ensure the submission columns are in the correct order

submission = submission[['Id', 'Fraction_Insertions', 'Avg_Deletion_Length', 'Indel_Diversity', 'Fraction_Frameshifts']]



# Display the first few rows of the submission to verify

print(submission.head())

# Save the submission file
submission.to_csv('path_to_your_submission.csv', index=False)
