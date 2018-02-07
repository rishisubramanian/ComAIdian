import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelBinarizer

class MultiLabelBinarizer:
    def __init__(self, columns=None):
        self.columns = columns
        self.binarizers = {}

    def binarize_column(self, input_frame, output_frame, column):
        data = input_frame[column]
        binarizer = self.binarizers[column]
        encoding = binarizer.transform(data)
        for i in range(encoding.shape[1]):
            column_name = binarizer.classes_[i]
            output_frame[column_name] = encoding[:,i]
        output_frame.drop(columns=[column], inplace=True)

    def fit(self, X):
        if self.columns is not None:
            for column in self.columns:
                data = X[column]
                binarizer = LabelBinarizer()
                binarizer.fit(data)
                self.binarizers[column] = binarizer
        else:
            for column in X.columns:
                data = X[column]
                binarizer = LabelBinarizer()
                binarizer.fit(data)
                self.binarizers[column] = binarizer

    def transform(self, X):
        output_df = X.copy()

        if self.columns is not None:
            for column in self.columns:
                self.binarize_column(X, output_df, column)
        else:
            for column in X.columns:
                self.binarize_column(X, output_df, column)
        return output_df

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
