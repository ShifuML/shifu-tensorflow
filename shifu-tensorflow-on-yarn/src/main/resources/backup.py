"""Synchronous SGD
Author: Tommy Mulc
"""

# from __future__ import print_function
import os
import tensorflow as tf
import argparse
import time
import sys
import logging
import gzip
from StringIO import StringIO
import random
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
import json

REPLICAS_TO_AGGREGATE_RATIO = 1
FEATURE_COUNT = 30
HIDDEN_NODES_COUNT = 20
VALID_TRAINING_DATA_RATIO = 0.3
DELIMITER = '|'
BATCH_SIZE = 10
EPOCH = 10 # TODO: should consider recovery from checkpoint, we need to reduce current global step
WORKING_DIR = "hdfs://horton/user/webai/.yarn/"

# read from env
cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
n_pss = len(cluster_spec['ps'])  # the number of parameter servers
n_workers = len(cluster_spec['worker'])  # the number of worker nodes
job_name = os.environ["JOB_NAME"]
task_index = int(os.environ["TASK_ID"])

logging.info("job_name:%s, task_index:%d" % (job_name, task_index))


def main(_):
    logging.getLogger().setLevel(logging.INFO)

    ps_hosts = cluster_spec['ps']
    worker_hosts = cluster_spec['worker']
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # allows this node know about all other nodes
    if job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=task_index)
        server.join()
    else:  # it must be a worker server
        logging.info("Loading data from worker index = %d" % task_index)

        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=task_index)

        logging.info("backup worker join!!")

        while True:
            time.sleep(1000000)



if __name__ == '__main__':
    tf.app.run()
