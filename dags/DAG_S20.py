from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook
import pandas as pd
import numpy as np
import boto3
from catboost import CatBoostRegressor
import logging
import io
from io import StringIO
import pickle
import os
from psycopg2.extras import execute_values
from prep_S20 import preprocess_data


S3_BUCKET = Variable.get("S3_BUCKET")
S3_ACCESS_KEY = Variable.get("S3_ACCESS_KEY")
S3_SECRET_KEY = Variable.get("S3_SECRET_KEY")
S3_MODEL_KEY = Variable.get("S3_MODEL_KEY")

POSTGRES_CONN_ID = "postgres_default"

logger = logging.getLogger(__name__)


def load_data_from_postgres(**context):
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # удаляем если существует 
    cursor.execute("DROP TABLE IF EXISTS inference_data_temp")
    
    # создаём обычную таблицу 
    create_table_query = """
        CREATE TABLE inference_data_temp AS
    SELECT 
        COALESCE(s.store, p.store) as store,
        COALESCE(s.dept, p.dept) as dept,
        COALESCE(s.date, p.date) as date,
        s.weekly_sales,
        COALESCE(s.is_holiday, p.is_holiday) as is_holiday,
        st.type,
        st.size,
        f.temperature,
        f.fuel_price,
        f.cpi,
        f.unemployment,
        f.factor1,
        f.factor2,
        f.factor3,
        f.factor4,
        f.factor5
    FROM plan p
    LEFT JOIN sales s ON p.store = s.store AND p.dept = s.dept AND p.date = s.date
    LEFT JOIN stores st ON COALESCE(s.store, p.store) = st.store
    LEFT JOIN features f ON COALESCE(s.date, p.date) = f.date
    """
    cursor.execute(create_table_query)
    conn.commit()
    
    # теперь читаем данные
    cursor.execute("SELECT * FROM inference_data_temp")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=colnames)
    
    logger.info(f"Строк: {len(df)}")
    logger.info(f"Мин дата: {df['date'].min()}")
    logger.info(f"Макс дата: {df['date'].max()}")
    
    # получаем первую дату плана
    cursor.execute("SELECT MIN(date) FROM plan")
    first_plan_date = cursor.fetchone()[0]
    context['task_instance'].xcom_push(key='first_plan_date', value=str(first_plan_date))
    
    cursor.close()
    conn.close()


def preprocess_features(**context):
    logger.info("=== preprocess_features START ===")
    first_plan_date = context['task_instance'].xcom_pull(key='first_plan_date', task_ids='load_data')
    logger.info(f"first_plan_date: {first_plan_date}")
    
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM inference_data_temp")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=colnames)
    logger.info(f"Initial df shape: {df.shape}")
    
    df = preprocess_data(df)
    logger.info(f"After preprocess_data shape: {df.shape}")
    
    df = df[df['date'] >= first_plan_date]
    logger.info(f"After date filter (>= {first_plan_date}) shape: {df.shape}")
    
    df = df.drop(columns=['weekly_sales'])
    
    # Заполняем нулями ТОЛЬКО лаговые колонки
    lag_columns = ['sales_1week_ago', 'sales_2week_ago', 'sales_4week_ago', 
                   'mean_sales_2week', 'mean_sales_4week', 'avg_sales_before']
    
    for col in lag_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # удаляем обычную таблицу
    cursor.execute("DROP TABLE IF EXISTS inference_data_temp")
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info(f"Final df shape: {df.shape}")
    logger.info(f"NaN check:\n{df.isnull().sum()}")
    
    # push в XCom
    context['task_instance'].xcom_push(key='inference_df', value=df.to_json())
    logger.info("=== preprocess_features END ===")


def load_model_from_s3(**context):
    s3_client = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name='ru-central1'
    )
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
        model_bytes = response['Body'].read()
        model = pickle.load(io.BytesIO(model_bytes))
        
        model_path = '/tmp/catboost_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # проверка что файл создан
        import os
        if os.path.exists(model_path):
            logger.info(f"Модель сохранена: {model_path}, размер: {os.path.getsize(model_path)}")
        else:
            logger.error("Файл модели не создан")
        
        context['task_instance'].xcom_push(key='model_path', value=model_path)
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели из S3: {e}")
        raise


def run_batch_inference(**context):
    logger.info("=== run_batch_inference START ===")
    
    inference_json = context['task_instance'].xcom_pull(key='inference_df', task_ids='preprocess_features')
    model_path = context['task_instance'].xcom_pull(key='model_path', task_ids='load_model')
    
    if model_path is None:
        raise ValueError("model_path не найден в XCom")
    
    df = pd.read_json(StringIO(inference_json))
    logger.info(f"df shape before predict: {df.shape}")
    
    # сохраняем для результата
    stores = df['store'].copy()
    dates = df['date'].copy()
    depts = df['dept'].copy()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # удаляем колонки, которых нет в модели
    features_df = df.drop(columns=['store', 'date'])
    
    # явно указываем категориальные колонки
    cat_features = ['dept', 'type', 'is_holiday', 'season']
    
    for col in cat_features:
        if col in features_df.columns:
            features_df[col] = features_df[col].astype(str)
    
    # проверка, что порядок колонок должен совпадать с обучением
    expected_features = model.feature_names_
    logger.info(f"Expected features: {expected_features}")
    logger.info(f"Actual features: {features_df.columns.tolist()}")
    
    # переставляем колонки в правильном порядке
    features_df = features_df[expected_features]
    
    features_df['predicted_weekly_sales'] = model.predict(features_df)
    features_df.loc[features_df['predicted_weekly_sales'] < 0, 'predicted_weekly_sales'] = 0
    
    result_df = pd.DataFrame({
        'store': stores,
        'dept': depts,
        'date': dates,
        'predicted_weekly_sales': features_df['predicted_weekly_sales']
    })
    
    context['task_instance'].xcom_push(key='predictions_df', value=result_df.to_json())
    logger.info("=== run_batch_inference END ===")

def save_predictions_to_postgres(**context):
    predictions_json = context['task_instance'].xcom_pull(key='predictions_df', task_ids='run_inference')
    predictions_df = pd.read_json(predictions_json)
    
    predictions_df['date'] = pd.to_datetime(predictions_df['date']).dt.date
    
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # убрали drop table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS predictions (
        store INT,
        dept INT,
        date DATE,
        predicted_weekly_sales FLOAT,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (store, dept, date)
    );
    """
    cursor.execute(create_table_query)
    conn.commit()

    
    values = predictions_df[['store', 'dept', 'date', 'predicted_weekly_sales']].to_records(index=False).tolist()
    
    insert_query = """
    INSERT INTO predictions (store, dept, date, predicted_weekly_sales)
    VALUES %s
    ON CONFLICT (store, dept, date) 
    DO UPDATE SET 
        predicted_weekly_sales = EXCLUDED.predicted_weekly_sales,
        prediction_timestamp = CURRENT_TIMESTAMP;
    """
    
    execute_values(cursor, insert_query, values)
    conn.commit()
    
    count_query = "SELECT COUNT(*) FROM predictions"
    cursor.execute(count_query)
    count = cursor.fetchone()[0]
    logger.info(f"Количество строк в таблице predictions: {count}")
    
    select_query = "SELECT * FROM predictions LIMIT 5"
    cursor.execute(select_query)
    rows = cursor.fetchall()
    logger.info("Первые 5 строк таблицы predictions:")
    for row in rows:
        logger.info(row)
    
    cursor.close()
    conn.close()


default_args = {
    'owner': 'Sergey Bazavluk',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'prilavok_batch_inference',
    default_args=default_args,
    description='Batch-инференс прогнозирования продаж для Прилавка',
    schedule_interval='0 20 * * 0',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['sales', 'ml', 'batch-inference', 'prilavok'],
    max_active_runs = 1
)

task_load_data = PythonOperator(
    task_id='load_data',
    python_callable=load_data_from_postgres,
    provide_context=True,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_features',
    python_callable=preprocess_features,
    provide_context=True,
    dag=dag,
)

task_load_model = PythonOperator(
    task_id='load_model',
    python_callable=load_model_from_s3,
    provide_context=True,
    dag=dag,
)

task_inference = PythonOperator(
    task_id='run_inference',
    python_callable=run_batch_inference,
    provide_context=True,
    dag=dag,
)

task_save_predictions = PythonOperator(
    task_id='save_predictions',
    python_callable=save_predictions_to_postgres,
    provide_context=True,
    dag=dag,
)

task_load_data >> task_preprocess
task_load_model >> task_inference
task_preprocess >> task_inference >> task_save_predictions