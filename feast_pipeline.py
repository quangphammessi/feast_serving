import os
import kfp.dsl as dsl
from kfp.dsl import PipelineVolume

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


def git_clone_op(repo_url: str):
    image='alpine/git:latest'
    
    commands = [
        f"git clone {repo_url} {PROJECT_ROOT}",
        f"cd {PROJECT_ROOT}"]

    volume_op = dsl.VolumeOp(
        name='create pipeline volume',
        resource_name='pipeline-pvc',
        modes=["ReadWriteOnce"],
        size='3Gi'
    )

    op = dsl.ContainerOp(
        name='git clone',
        image=image,
        command=['sh'],
        arguments=['-c', ' && '.join(commands)],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={WORKSPACE: volume_op.volume}
    )

    return op

def fetch_feast_feature(image: str, pvolume: PipelineVolume):
    op = dsl.ContainerOp(
        name='feature_store',
        image=image,
        command=[CONDA_PYTHON_CMD, f'{PROJECT_ROOT}/create_feature_store.py'],
        container_kwargs={'image_pull_policy': 'IfNotPresent'},
        pvolumes={WORKSPACE: pvolume}
    )


@dsl.pipeline(
    name='Feast Image Pipeline',
    description='Feast Image Pipeline to be executed on KubeFlow.'
)
def training_pipeline(image: str='quangphammessi/feast_serving:latest',
                        repo_url: str='https://github.com/quangphammessi/feast_serving',
                        data_dir: str='/workspace'):
    git_clone = git_clone_op(repo_url=repo_url)

    fetch_feast = fetch_feast_feature(image=image, pvolume=git_clone.pvolume)

if __name__ == '__main__':
    import kfp.compiler as compiler

    compiler.Compiler().compile(training_pipeline, __file__ + '.tar.gz')