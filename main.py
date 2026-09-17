import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

class LinearRegressionModel:

    learning_rate = 1e-2
    slope_weights = np.random.rand(2)
    intercept_weight = np.random.rand(1)

    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate
        self.mode_type = type

    def r2_score(self, y_true, y_pred):
        err2 = np.sum((y_true - y_pred) ** 2)
        var = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (err2 / var)

    def gradient_descent(self, X, y, iterations):
        N = len(y)
        loss = []
        curr_r2 = 0
        # for _ in range(iterations):
        while curr_r2 <= 0:
            predictions = self.predict(X)
            errors = y - predictions 
            dL_dm = -(2/N) * np.dot(X.T, errors)
            dL_db = -(2/N) * np.sum(errors)

            self.slope_weights -= self.learning_rate * dL_dm
            self.intercept_weight -= self.learning_rate * dL_db

            curr_r2 = self.r2_score(y,predictions)
            loss.append(np.dot(errors,errors)/N)

        fig, ax = plt.subplots()
        ax.plot(loss)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('MSE')
        ax.set(title=f'MSE Loss on Training Data For Multivariate, Log-Transformed Model (lr={self.learning_rate})')
        plt.show()

    def train(self, X, y, iterations=1000):
        self.slope_weights = np.random.rand(X.shape[1])
        self.intercept_weight = np.random.rand(1)
        self.gradient_descent(X, y, iterations=iterations)

    def predict(self, X):
        return np.dot(X, self.slope_weights) + self.intercept_weight

    def test(self, train, test):

        x_train = train[:,:-1].reshape(len(train),-1)
        y_train = train[:,-1]
        predict_train = self.predict(x_train)
        errors_train = y_train - predict_train
        mse_train = np.dot(errors_train,errors_train)/len(y_train)
        r2_train = self.r2_score(y_train, predict_train)

        x_test = test[:,:-1].reshape(len(test),-1)
        y_test = test[:,-1]
        predict_test = self.predict(x_test)
        errors_test = y_test - predict_test
        mse_test = np.dot(errors_test,errors_test)/len(y_test)
        r2_test = self.r2_score(y_test, predict_test)

        return mse_train, r2_train, mse_test, r2_test

    def summarize(self, train, test):
        mse_train, r2_train, mse_test, r2_test = self.test(train,test)
        print("Gradient Descent Summary:")
        print(f"Train: MSE={mse_train}, R2 Score={r2_train}")
        print(f"Test: MSE={mse_test}, R2 Score={r2_test}\n")

    def print_weights(self):
        print(f"Weights: slope={self.slope_weights}, intercept={self.intercept_weight}")

def import_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
        return None
    except pd.errors.ParserError:
        print("Error parsing the file.")
        return None

def train_test_split(df):
    indices = np.arange(df.shape[0])
    test_indices = (indices >= 501) & (indices <= 630)
    test = df.loc[test_indices,:]
    train = df.loc[~test_indices,:]
    return np.array(train), np.array(test)

def min_max_normalize(data):
    col_min = data.min(axis=0, keepdims=True)
    col_max = data.max(axis=0, keepdims=True)
    norm_data = (data - col_min) / (col_max - col_min)
    return norm_data

def log_transform(data):
    return np.log(data + 1)


def run_gradient_descent(train,test):
    x = train[:, :-1].reshape((len(train),-1))
    y = train[:, -1]
    model = LinearRegressionModel(learning_rate=1e-6)
    model.train(x, y, iterations=10000)
    model.summarize(train, test)
    # model.print_weights()

def run_sk_linear_regression(train,test):
    x = train[:, :-1].reshape((len(train),-1))
    y = train[:, -1]

    model = LinearRegression().fit(x,y)
    score_train = model.score(x,y)
    score_test = model.score(test[:,:-1],test[:,-1])
    slopes = model.coef_
    intercept = model.intercept_
    train_preds = np.dot(train[:,:-1],slopes) + intercept
    errors_train = train[:,-1] - train_preds
    mse_train = np.dot(errors_train,errors_train)/len(train[:,-1])
    print("SK Linear Regression Summary")
    print(f"Train: MSE = {mse_train}, R2 = {score_train}",)

    test_preds = np.dot(test[:,:-1],slopes) + intercept
    errors_test = test[:,-1] - test_preds
    mse_test = np.dot(errors_test,errors_test)/len(test[:,-1])
    print(f"Test: MSE = {mse_test}, R2 = {score_test}\n")
    # print("Weights + Intercept: ", slopes,intercept)

def run_statsmodels_ols(data):
    X = sm.add_constant(data[data.columns[:-1]])
    Y = data["Concrete compressive strength(MPa, megapascals) "]
    model = sm.OLS(Y,X).fit()
    print("Statsmodels OLS P-values")
    print(model.pvalues)

def __main__():

    np.random.seed(0)
    data = import_data('Concrete_Data.csv')
    train, test = train_test_split(data)

    ### Univariate modeling 
    # col_to_keep = 2
    # train = train[:,[col_to_keep,-1]]
    # test = test[:,[col_to_keep,-1]]

    ### Transform features 
    # train = min_max_normalize(train)
    train = log_transform(train)

    ### Run algorithms 
    run_gradient_descent(train,test)
    run_sk_linear_regression(train,test)
    run_statsmodels_ols(data)


if __name__ == "__main__":
    __main__()