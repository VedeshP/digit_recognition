from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

y = y.astype('int64')

X = X.values.reshape(-1, 28, 28)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

def load_my_data():
    return X_train, X_test, y_train, y_test

# print(y.dtype)
print(y_train.dtype)
print(X.shape)