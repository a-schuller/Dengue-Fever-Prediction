import numpy as np
import pandas as pd

class ProcessingData:

    def duplicates_drop(df):
        df.drop_duplicates(inplace=True)

    def fill_data(df, fillType):
        if fillType == 'ffill':
            return df.fillna(method='ffill', inplace=True)
        elif fillType == 'fmean':
            return df.fillna(df.mean(), inplace=True)
        else:
            return df.fillna(0, inplace=True)

    def drop_outlier(df, df_test, to_trim):
        for v in to_trim:
            df.loc[:,v] = [min(x, df_test[v].max()) for x in df[v]]
            df.loc[:,v] = [max(x, df_test[v].min()) for x in df[v]]

    def city_split(self, splist):
        # separate San Juan and Iquitos
        self.ifsplit = splist
        if self.ifsplit:
            self.sj = self.df[self.df.loc[:, 'city'] == 'sj']
            self.iq = self.df[self.df.loc[:, 'city'] == 'iq']
            return self.sj, self.iq
        else:
            return self.df

    def cyclical_encoding(data: pd.DataFrame, column: str) -> pd.DataFrame:
        new_column_name_sin = f"sin_{column}"
        new_column_name_cos = f"cos_{column}"

        data.loc[:, new_column_name_sin] = np.sin(2 * np.pi * data.loc[:, column] / max(data.loc[:, column]))
        data.loc[:, new_column_name_cos] = np.cos(2 * np.pi * data.loc[:, column] / max(data.loc[:, column]))
        data.drop(columns=column, inplace=True)

        return data