import os

from feast import Client, Feature, Entity, ValueType, FeatureTable, client
from feast.data_source import FileSource, KafkaSource
from feast.data_format import ParquetFormat, AvroFormat

import numpy as np
import cv2
import random

import pandas as pd
from datetime import datetime


SAVE_STEPS = 1000
VALIDATE_EPOCHS = 10

BATCH_SIZE = 64
BATCH_PER_EPOCH = 50

CHECKPOINT_DIR = 'checkpoint'

IMG_SIZE = [94, 24]
CH_NUM = 3

CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789" # exclude I, O
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
DECODE_DICT = {i:char for i, char in enumerate(CHARS)}

NUM_CLASS = len(CHARS)+1


def encode_label(label, char_dict):
    encode = [char_dict[c] for c in label]
    return encode

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


class DataIterator:
    def __init__(self, img_dir, runtime_generate=False):
        self.img_dir = img_dir
        self.batch_size = BATCH_SIZE
        self.channel_num = CH_NUM
        self.img_w, self.img_h = IMG_SIZE

        self.init()

    def init(self):
        self.filenames = []
        self.labels = []
        fs = os.listdir(self.img_dir)
        for filename in fs:
            self.filenames.append(filename)
            label = filename.split('_')[0] # format: [label]_[random number].jpg
            label = encode_label(label, CHARS_DICT)
            self.labels.append(label)
        self.sample_num = len(self.labels)
        self.labels = np.array(self.labels)
        self.random_index = list(range(self.sample_num))
        random.shuffle(self.random_index)
        self.cur_index = 0

    def next_sample_ind(self):
        ret = self.random_index[self.cur_index]
        self.cur_index += 1
        if self.cur_index >= self.sample_num:
            self.cur_index = 0
            random.shuffle(self.random_index)
        return ret

    def next_batch(self):

        batch_size = self.batch_size
        images = np.zeros([batch_size, self.img_h, self.img_w, self.channel_num])
        labels = []

        for i in range(batch_size):
            sample_ind = self.next_sample_ind()
            fname = self.filenames[sample_ind]
            img = cv2.imread(os.path.join(self.img_dir, fname))
            #img = data_augmentation(img)
            img = cv2.resize(img, (self.img_w, self.img_h))
            images[i] = img

            labels.append(self.labels[sample_ind])

        sparse_labels = sparse_tuple_from(labels)

        return images, sparse_labels, labels

    def next_test_batch(self):

        start = 0
        end = self.batch_size
        is_last_batch = False

        while not is_last_batch:
            if end >= self.sample_num:
                end = self.sample_num
                is_last_batch = True

            #print("s: {} e: {}".format(start, end))

            cur_batch_size = end-start
            images = np.zeros([cur_batch_size, self.img_h, self.img_w, self.channel_num])

            for j, i in enumerate(range(start, end)):
                fname = self.filenames[i]
                img = cv2.imread(os.path.join(self.img_dir, fname))
                img = cv2.resize(img, (self.img_w, self.img_h))
                images[j, ...] = img

            labels = self.labels[start:end, ...]
            sparse_labels = sparse_tuple_from(labels)

            start = end
            end += self.batch_size

            yield images, sparse_labels, labels

    def next_gen_batch(self):

        batch_size = self.batch_size
        imgs, labels = self.generator.generate_images(batch_size)
        labels = [encode_label(label, CHARS_DICT) for label in labels]

        images = np.zeros([batch_size, self.img_h, self.img_w, self.channel_num])
        for i, img in enumerate(imgs):
            img = data_augmentation(img)
            img = cv2.resize(img, (self.img_w, self.img_h))
            images[i, ...] = img

        sparse_labels = sparse_tuple_from(labels)

        return images, sparse_labels, labels

    
def create_feature_serving():
    cur_path = os.getcwd()

    staging_bucket = f'file://{cur_path}'

    environment = {
    #               'FEAST_CORE_URL': 'feast-release-feast-core.default:6565',
        'FEAST_CORE_URL': 'localhost:6565',
    #               'FEAST_DATAPROC_CLUSTER_NAME': 'dataprocfeast',
    #               'FEAST_DATAPROC_PROJECT': '<BUCKET>',
    #               'FEAST_DATAPROC_REGION': 'us-east1',
                    'FEAST_STAGING_LOCATION': staging_bucket,
                    'FEAST_HISTORICAL_FEATURE_OUTPUT_FORMAT': 'parquet',
                    'FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION': f"{staging_bucket}historical" ,
    #               'FEAST_HISTORICAL_SERVING_URL': 'feast-release-feast-serving.default:6566',
        'FEAST_HISTORICAL_SERVING_URL': 'localhost:6566',
                    'FEAST_REDIS_HOST': '<REDIS_IP>',
                    'FEAST_REDIS_PORT': '6379',
    #               'FEAST_SERVING_URL': 'feast-release-feast-serving.default:6566',
        'FEAST_SERVING_URL': 'localhost:6566',
                    'FEAST_SPARK_HOME': '/usr/local/spark',
                    'FEAST_SPARK_LAUNCHER': 'standalone',
                    'FEAST_SPARK_STAGING_LOCATION': f'{staging_bucket}/spark_staging_location/',
                    'FEAST_SPARK_STANDALONE_MASTER': 'local[*]',
                    'STAGING_BUCKET': 'self.staging_bucket',
                    'DEMO_KAFKA_BROKERS': '<KAFKA_IP>'
                    
                    }

    # Define Client
    client = Client()
    
    # Get data
    train_gen = DataIterator(img_dir='tensorflow_lprnet_test/train')

    # Create Entity and Feature
    image_id = Entity(name='image_id', description='Image ID', value_type=ValueType.INT64)

    image_value_batch = Feature('image_value_batch', ValueType.INT64)
    image_value_row = Feature('image_value_row', ValueType.INT64)
    image_value_col = Feature('image_value_col', ValueType.INT64)

    image_value_r = Feature('image_value_r', ValueType.DOUBLE)
    image_value_g = Feature('image_value_g', ValueType.DOUBLE)
    image_value_b = Feature('image_value_b', ValueType.DOUBLE)
    label = Feature('label', ValueType.INT64_LIST)

    data_location = os.path.join(os.getenv('FEAST_SPARK_STAGING_LOCATION', staging_bucket), 'test_data')

    image_value_feature_source_uri = os.path.join(data_location, 'image_value_feature')

    # Create Feature Table
    image_value_feature = FeatureTable(
        name='image_value_feature',
        entities=['image_id'],
        features=[
            image_value_batch,
            image_value_row,
            image_value_col,
            image_value_r,
            image_value_g,
            image_value_b
        ],
        batch_source=FileSource(
            event_timestamp_column='datetime',
            created_timestamp_column='created',
            file_format=ParquetFormat(),
            file_url=image_value_feature_source_uri
        )
    )

    client.apply(image_id)
    client.apply(image_value_feature)


    # Populating batch source
    pre_data = train_gen.next_batch()
    pre_images = pre_data[0]
    pre_label = pre_data[2]

    index = pd.MultiIndex.from_product(
        (*map(range, pre_images.shape[:3]), ('image_value_r', 'image_value_g', 'image_value_b')),
        names=('image_value_batch', 'image_value_row', 'image_value_col', None)
    )

    out_images = pd.Series(pre_images.flatten(), index=index).unstack().reset_index()
    entities = np.random.choice(999999, size=out_images.shape[0], replace=False)
    out_images['image_id'] = entities

    out_images['datetime'] = pd.to_datetime(
                np.random.randint(
                    datetime(2021, 7, 1).timestamp(),
                    datetime(2021, 7, 31).timestamp(),
                    size=out_images.shape[0]),
            unit="s"
        )
    out_images['created'] = pd.to_datetime(datetime.now())

    client.ingest(image_value_feature, out_images)