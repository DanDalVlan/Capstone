import os
import sys

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm, model_selection, metrics
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Get absolute path for pyinstaller to find the csv file
def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Importing data
csv_file_path = resource_path('data_final_no_headers.csv')
names = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
         'Symmetry', 'Fractal Dimension', 'Diagnosis']
df = pd.read_csv(csv_file_path, names=names)

# Define the dependent and independent variables
X = df[['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
        'Symmetry', 'Fractal Dimension']].values
y = df['Diagnosis'].values
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# Create a pipeline that scales the data and then fits the SVM model
mysvm_model = make_pipeline(StandardScaler(), svm.SVC(max_iter=1000))

# Train the model
mysvm_model.fit(X_train, y_train)

# Predictions
y_pred_svm = mysvm_model.predict(X_test)


# Function to ensure valid user input
def get_valid_input(prompt, min_value, max_value):
    while True:
        user_input = input(prompt)
        try:
            user_input = float(user_input)
            if min_value <= user_input <= max_value:
                return user_input
            else:
                print(f"Please enter a number between {min_value} and {max_value}")
        except ValueError:
            print("Please enter an integer")


# Function that creates and shows the graph of characteristics and their importance to the model
def importance_graph():
    # Create a temp linear SVC model for this graph
    svm_graph_model = SVC(kernel='linear', random_state=0)
    svm_graph_model.fit(X_train, y_train)
    # Sort the temp model and assign importance
    tumor_list = df.columns[:-1]
    list_importance = svm_graph_model.coef_[0]
    sorted_list = np.argsort(np.abs(list_importance))
    # Create and display the graph
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_list)), np.abs(list_importance[sorted_list]), color="blue")
    plt.yticks(range(len(sorted_list)), tumor_list[sorted_list])
    plt.title('The Importance of Each Tumor Characteristic to the Model')
    plt.xlabel('Importance')
    plt.ylabel('Characteristics')
    plt.show()


# Function that creates and shows a pair plot
def pairplot_graph():
    sns.pairplot(df, hue='Diagnosis', markers=["o", "s"], palette="Set1")
    plt.suptitle('Pair Plot of Tumor Characteristics by Diagnosis', size=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


# Function that creates and shows a ROC Curve
def roc_graph():
    RocCurveDisplay.from_estimator(mysvm_model, X_test, y_test)
    plt.title('ROC Curve')
    plt.show()


# Main Function that contains all user interactions
class Main:
    # Print hello plus general info
    print("\nHello and welcome to the McGuffin Institute's Tumor Diagnostic Project")
    print("You will be prompted for the following 10 tumor characteristic values, please ensure you have them on hand:")
    print("1. Radius\n2. Texture\n3. Perimeter\n4. Area\n5. Smoothness\n6. Compactness\n7. Concavity\n8. Concave points"
          "\n9. Symmetry\n10. Fractal Dimension\n")

    # Loop to let the user interact with the menu until they choose to end
    input_selection = 0
    while True:
        input_selection = get_valid_input("\nSelect an option:\n1. Begin Diagnostics\n2. Verify Methodology\n"
                                          "3. End Diagnostics\n", 1, 3)

        # If 1, prompt them for all the inputs and show prediction
        if input_selection == 1:
            # Receive inputs from the users
            radius = get_valid_input("1. Radius in mm (mean of distances from center to points on the "
                                     "perimeter): ", 0, 50)
            texture = get_valid_input("2. Texture (standard deviation of gray-scale values): ",
                                      0, 50)
            perimeter = get_valid_input("3. Perimeter in mm: ", 0, 300)
            area = get_valid_input("4. Area in mm: ", 0, 5000)
            smoothness = get_valid_input("5. Smoothness (local variation in radius lengths): ",
                                         0, 1)
            compactness = get_valid_input("6. Compactness (perimeter^2 / area - 1.0): ", 0, 1)
            concavity = get_valid_input("7. Concavity (severity of concave portions of the contour): ",
                                        0, 1)
            concave_points = get_valid_input("8. Concave Points (number of concave portions of the contour): ",
                                             0, 1)
            symmetry = get_valid_input("9. Symmetry: ", 0, 1)
            fractal_dimension = get_valid_input("10. Fractal Dimensions ('coastline approximation' - 1): ",
                                                0, 1)

            # Creating and formatting the prediction
            prediction_base = mysvm_model.predict([[radius, texture, perimeter, area, smoothness, compactness,
                                                    concavity, concave_points, symmetry, fractal_dimension]])
            if prediction_base == 'M':
                prediction = 'Malignant'
            else:
                prediction = 'Benign'

            # Formatting and printing the prediction/accuracy
            accuracy = metrics.accuracy_score(y_test, y_pred_svm) * 100
            print("\nOur diagnostic prediction for the tumor whose characteristics were provided is: ", prediction)
            print("This model has an accuracy of: {:.2f}%".format(accuracy))

        # Else if 2, show next menu for visualizations
        elif input_selection == 2:
            # Loop to let the user choose which visuals they want to see, or end loop
            while True:
                input_visuals = get_valid_input("Which visualization would you like to look at?\n"
                                                "1. Importance Graph\n2. Pair Plot\n3. ROC Curve\n"
                                                "4. Return to previous menu\n", 1, 4)
                # If 1, call function that creates and displays the importance graph
                if input_visuals == 1:
                    importance_graph()
                # Else if 2, call function that creates and displays pairplot graph
                elif input_visuals == 2:
                    pairplot_graph()
                # Else if 3, call function that creates and displays ROC curve
                elif input_visuals == 3:
                    roc_graph()
                # Else if 4, end loop
                elif input_visuals == 4:
                    break
                # Else tell user their selection is invalid
                else:
                    print("Invalid selection")

        # Else if 3, break loop
        elif input_selection == 3:
            break

        # Else tell user their selection is invalid
        else:
            print("Invalid selection")
