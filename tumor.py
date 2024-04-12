import pandas as pd
from sklearn import metrics, model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load your data
url = "https://raw.githubusercontent.com/DanDalVlan/Capstone/working/capstone_data_short.csv"
names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
         'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
         'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
         'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
         'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
         'fractal_dimension_worst', 'diagnosis']
df = pd.read_csv(url, names=names)

# Define the features and target
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
        'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
        'fractal_dimension_worst']].values
y = df['diagnosis'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# Create a pipeline that scales the data and then fits the logistic regression model
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
# Here we need to scale the input features in the same way as the training data was scaled
# prediction = pipeline.predict([[3, 5, 5]])
# print(prediction)

y_pred = pipeline.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
