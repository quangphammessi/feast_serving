import cv2
import os
import time
import json
import random
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from feast.config import Config
from feast.data_source import FileSource, KafkaSource
from feast.data_format import ParquetFormat, AvroFormat
from feast import Client, Feature, Entity, ValueType, FeatureTable, client
from feast.pyspark.abc import RetrievalJobParameters, SparkJobStatus, SparkJob
import feast.staging.entities as entities

import gcsfs
from pyarrow.parquet import ParquetDataset
from urllib.parse import urlparse

# To compile the pipeline:
#   dsl-compile --py pipeline.py --output pipeline.tar.gz

WORKSPACE = '/workspace'
PROJECT_ROOT = os.path.join(WORKSPACE, 'feast_serving')
CONDA_PYTHON_CMD = '/opt/conda/envs/kubeflow-lpr/bin/python'

class feature_store_client:
    
    def __init__(self,env,bucket):
        
        self.env=env
        self.staging_bucket=bucket
        
    def feature_store_settings(self):
        
        if self.env.lower()=="dataproc":
            # Using environmental variables
            environment = {
                'FEAST_CORE_URL': 'feast-release-feast-core.default.svc:6565',
                # 'FEAST_CORE_URL': 'localhost:6565',
#                          'FEAST_DATAPROC_CLUSTER_NAME': 'dataprocfeast',
#                          'FEAST_DATAPROC_PROJECT': '<BUCKET>',
#                          'FEAST_DATAPROC_REGION': 'us-east1',
                        #  'FEAST_STAGING_LOCATION': self.staging_bucket,
                         'FEAST_HISTORICAL_FEATURE_OUTPUT_FORMAT': 'parquet',
                        #  'FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION': f"{self.staging_bucket}/historical_feature_output" ,
                         'FEAST_HISTORICAL_SERVING_URL': 'feast-release-feast-serving.default.svc:6566',
                # 'FEAST_HISTORICAL_SERVING_URL': 'localhost:6566',
                         'FEAST_REDIS_HOST': 'feast-release-redis-headless.default.svc',
                         'FEAST_REDIS_PORT': '6379',
                         'FEAST_SERVING_URL': 'feast-release-feast-serving.default.svc:6566',
                # 'FEAST_SERVING_URL': 'localhost:6566',
                         'FEAST_SPARK_HOME': '/usr/local/spark',
                        #  'FEAST_SPARK_LAUNCHER': 'standalone',
                         'FEAST_SPARK_LAUNCHER': 'k8s',
                         'SPARK_K8S_NAMESPACE': 'kubeflow',
                        #  'FEAST_SPARK_STAGING_LOCATION': f'{self.staging_bucket}/spark_staging_location/',
                        #  'FEAST_SPARK_STANDALONE_MASTER': 'local[*]',
                        #  'STAGING_BUCKET': f'{self.staging_bucket}',
                         'DEMO_KAFKA_BROKERS': 'feast-release-kafka-headless.default.svc'
                           
                          }              

            for key,value in environment.items():
                os.environ[key] = value

def create_staging_bucket():
    cur_path = os.getcwd()
    
    staging_bucket = f'file://{cur_path}'
#     staging_bucket = f'gs://feast-staging-bucket-{random.randint(1000000, 10000000)}/'
#     !gsutil mb {staging_bucket}
    print(f'Staging bucket is {staging_bucket}')
    return staging_bucket

def create_all():
    staging_bucket = create_staging_bucket()
    # staging_bucket = 'file:///home/jovyan/'
    set_env=feature_store_client('Dataproc',staging_bucket)
    set_env.feature_store_settings()

    client = Client()

    def read_parquet(uri):
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme == "file":
            return pd.read_parquet(parsed_uri.path)
        elif parsed_uri.scheme == "gs":
            fs = gcsfs.GCSFileSystem()
            files = ["gs://" + path for path in fs.glob(uri + '/part-*')]
            ds = ParquetDataset(files, filesystem=fs)
            return ds.read().to_pandas()
        elif parsed_uri.scheme == 's3':
            import s3fs
            fs = s3fs.S3FileSystem()
            files = ["s3://" + path for path in fs.glob(uri + '/part-*')]
            ds = ParquetDataset(files, filesystem=fs)
            return ds.read().to_pandas()
        elif parsed_uri.scheme == 'wasbs':
            import adlfs
            fs = adlfs.AzureBlobFileSystem(
                account_name=os.getenv('FEAST_AZURE_BLOB_ACCOUNT_NAME'), account_key=os.getenv('FEAST_AZURE_BLOB_ACCOUNT_ACCESS_KEY')
            )
            uripath = parsed_uri.username + parsed_uri.path
            files = fs.glob(uripath + '/part-*')
            ds = ParquetDataset(files, filesystem=fs)
            return ds.read().to_pandas()
        else:
            raise ValueError(f"Unsupported URL scheme {uri}")

    def wait_for_job_status(
        job: SparkJob,
        expected_status: SparkJobStatus,
        max_retry: int = 4,
        retry_interval: int = 5,
    ):
        for i in range(max_retry):
            if job.get_status() == expected_status:
                print("The Spark Job is Completed")
                return
            time.sleep(retry_interval)
        raise ValueError(f"Timeout waiting for job status to become {expected_status.name}")

    entities = np.random.choice(999999, size=144384, replace=False)

    entities_with_timestamp = pd.DataFrame(columns=['image_id', 'event_timestamp'])
    entities_with_timestamp['image_id'] = np.random.choice(entities, 100, replace=False)
    entities_with_timestamp['event_timestamp'] = pd.to_datetime(np.random.randint(
        datetime(2021, 7, 18).timestamp(),
        datetime(2021, 7, 20).timestamp(),
        size=100), unit='s')

    job = client.get_historical_features(
        feature_refs=[
            'image_value_feature:image_value_r',
            'image_value_feature:image_value_g',
            'image_value_feature:image_value_b'
        ],
        entity_source=entities_with_timestamp
    )

    features_outcome = pd.DataFrame()
    if True:
        output_file_uri = job.get_output_file_uri()
        wait_for_job_status(job,SparkJobStatus.COMPLETED)
        features_outcome = read_parquet(output_file_uri)

    print(features_outcome.head(5))

if __name__ == '__main__':
    create_all()