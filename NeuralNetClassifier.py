from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing
import pandas as pd

# Load in data
data = pd.read_csv("dataset.csv")
if not data.empty:
    print('Data loaded in')
    print('################')
    print()

# Set the data set columns
X = data.values[:, 0:56] 	# Training data between columns 0 and 56 - Total 57 columns
Y = data.values[:,57]		# Results 0 or 1 / not-spam or spam

# Re-Scale the training data
scaler = preprocessing.MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# Training test split of the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2)

mlp = MLPClassifier(hidden_layer_sizes=(45, 14),
                    learning_rate_init=0.002,
                    random_state=1,
                    tol=1e-7,
                    activation="tanh",
					alpha=0.02,
                    max_iter=10000,
                    verbose=True,
                    n_iter_no_change=15)

# Train
mlp = mlp.fit(X_train, Y_train)

# Set preditions based on X_test set
Y_prediction = mlp.predict(X_test)
print('################')
print()
print("Accuracy:",accuracy_score(Y_test,Y_prediction))


print("\nRunning Cross-Fold validation")

#Split the dataset for cross validation testing
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)


scores = cross_val_score(mlp, X, Y, cv=cv) # Score from cross validation
print("\nCross fold validation accuracy scores:",scores)
print("\nCross fold validation accuracy mean:",scores.mean())