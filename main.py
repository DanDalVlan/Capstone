import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model

# importing raw data
url = "https://raw.githubusercontent.com/DanDalVlan/Capstone/working/muse_v3_final.csv"
names = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
         'Symmetry', 'Fractal Dimension', 'Diagnosis']
df = pd.read_csv(url, names=names)

url = "https://raw.githubusercontent.com/DanDalVlan/Capstone/working/capstone_data_short.csv"
names = ['Valence', 'Arousal', 'Dominance', 'Track']
df = pd.read_csv(url, names=names)

# creating a pyplot
# df.hist()
# pyplot.show()

# creating a scatter matrix
# scatter_matrix(df)

# creating a model
mylog_model = linear_model.LogisticRegression()
y = df.values[:, 3]
x = df.values[:, 0:3]


# training the model
mylog_model.fit(x, y)

# making predictions
print(mylog_model.predict([[3, 5, 5]]))
