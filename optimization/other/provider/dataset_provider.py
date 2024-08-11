import pandas as pd

class DatasetProvider:
    @staticmethod
    def get_places():
        return pd.read_csv('./Optimization/other/Dataset/places.csv')

    @staticmethod
    def get_time_matrix():
        return pd.read_csv('./Optimization/other/Dataset/place_timematrix.csv')

    @staticmethod
    def get_schedule():
        df_schedule = pd.read_csv('./Optimization/other/Dataset/place_jadwal.csv')
        df_schedule['jam_buka'] = df_schedule['jam_buka'].apply(lambda x: int(x[:2]) * 3600 + int(x[3:]) * 60)
        df_schedule['jam_tutup'] = df_schedule['jam_tutup'].apply(lambda x: int(x[:2]) * 3600 + int(x[3:]) * 60)
        return df_schedule
