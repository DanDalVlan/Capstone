import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import linear_model

# importing raw data
url = "https://raw.githubusercontent.com/DanDalVlan/Capstone/working/muse_v3_final.csv"
names = ['Track', 'Artist', 'Genre', 'Valence', 'Arousal', 'Dominance']
df = pd.read_csv(url, names=names)

# creating a pyplot
#df.hist()
#pyplot.show()

# creating a scatter matrix
# scatter_matrix(df)

# creating a model
mylog_model = linear_model.LogisticRegression()
x = df[['Valence', 'Arousal', 'Dominance']]
y = df[:0]

# training the model
mylog_model.fit(x, y)

# making predictions
print(mylog_model.predict([[2.97, 5.936, 5.214]]))
