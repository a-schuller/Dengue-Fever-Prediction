"""Helper functions"""
import pandas as pd
import numpy as np
import datetime


def data_loader():
    df_features = pd.read_csv("./data/dengue_features_train.csv")
    df_test = pd.read_csv("./data/dengue_features_test.csv")
    df_label = pd.read_csv("./data/dengue_labels_train.csv")
    df_train = pd.concat([df_features, df_label.loc[:,'total_cases']], axis=1)
    df_train_features = pd.read_csv("./data/dengue_features_train.csv")
    df_label = pd.read_csv("./data/dengue_labels_train.csv")

    return df_train, df_test, df_train_features, df_label

def get_data_into_submission_format(prediction: np.array) -> None:
    """
    This function is used to put the predictions into the right
    submission format

    Args:
        prediction (np.array): This is an array with the predictions
    """
    subm = pd.read_csv('data/submission_format.csv')
    labels = pd.DataFrame({'total_cases': pd.Series(prediction)})
    subm.loc[:, 'total_cases'] = labels.astype(int)
    subm = subm.loc[:,["city", "year", "weekofyear", 'total_cases']]
    time = datetime.datetime.today().strftime("%Y_%m_%d_%H_%M")
    # write predictions into submission file
    subm.to_csv(f'./submission/submission_{time}.csv', index=False)