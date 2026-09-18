import pandas as pd
import numpy as np

def create_temporal_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week.astype('int64')
    df['quarter'] = df['date'].dt.quarter
    df['season'] = df['month'].map({12: 'Зима', 1: 'Зима', 2: 'Зима',
                                     3: 'Весна', 4: 'Весна', 5: 'Весна',
                                     6: 'Лето', 7: 'Лето', 8: 'Лето',
                                     9: 'Осень', 10: 'Осень', 11: 'Осень'})
    return df

def create_avg_sales_feature(my_df):
    df_featured = my_df.copy()
    
    # Сортируем данные для корректного окна расширения - сразу по магазину-департаменту-потом 
    df_featured = df_featured.sort_values(['store', 'dept', 'date'])
    
    # Вычисляем скользящее среднее по историческим данным
    df_featured['avg_sales_before'] = (
        df_featured
        .groupby(['store', 'dept'])['weekly_sales']
        .transform(lambda x: x.expanding().mean().shift(1)) #векторная операция для ускорения
        #убираем reset_index, т.к. значения признака присваиваются по позиции а не по исходному индексу
    )
    
    return df_featured

def create_lag_features(df):
    df_sales = df.copy()
    df_sorted = df_sales.sort_values('date')
    
    df_sales['sales_1week_ago'] = (
        df_sorted.groupby(['store', 'dept'])['weekly_sales']
        .shift(1).reset_index(drop=True)
    )
    df_sales['sales_2week_ago'] = (
        df_sorted.groupby(['store', 'dept'])['weekly_sales']
        .shift(2).reset_index(drop=True)
    )
    df_sales['sales_4week_ago'] = (
        df_sorted.groupby(['store', 'dept'])['weekly_sales']
        .shift(4).reset_index(drop=True)
    )
    return df_sales

def create_rolling_features(df):
    df_mean = df.copy()
    df_sorted = df_mean.sort_values('date')
    
    df_mean['mean_sales_2week'] = (
        df_sorted.groupby(['store', 'dept'])['weekly_sales']
        .shift(1).rolling(2, min_periods=2).mean()
        .reset_index(drop=True)
    )
    df_mean['mean_sales_4week'] = (
        df_sorted.groupby(['store', 'dept'])['weekly_sales']
        .shift(1).rolling(4, min_periods=4).mean()
        .reset_index(drop=True)
    )
    return df_mean

def preprocess_data(df):
    df = df.copy()
    
    # обрабатываем аномальные продажи
    if 'weekly_sales' in df.columns:
        df.loc[df['weekly_sales'] < 0, 'weekly_sales'] = 0
    
    # заполняем пропуски средним
    for col in ['factor2', 'factor3', 'factor4', 'factor5']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # конвертация температуры
    if 'temperature' in df.columns:
        df['temperature_c'] = (df['temperature'] - 32) * 5 / 9    
        df = df.drop('temperature', axis = 1)
    
    # доп признаки
    df = create_temporal_features(df)
    df = create_avg_sales_feature(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    return df