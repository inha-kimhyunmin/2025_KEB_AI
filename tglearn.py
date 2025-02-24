import numpy as np


class LinearRegression:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, x, y):
        """
        learning function
        :param x: independent variable
        :param y: dependent varianble
        :return: void
        """

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        denominator = np.sum(pow(x - x_mean, 2))
        numerator = np.sum((x - x_mean) * (y - y_mean))

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * x_mean)


    def predict(self, x) ->list:
        """
        predict value for input
        :param x: new independent variable
        :return: predict value for input(2d array format)
        """
        return self.slope * np.array(x) + self.intercept