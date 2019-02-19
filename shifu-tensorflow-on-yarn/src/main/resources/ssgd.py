"""Synchronous SGD
"""

#from __future__ import print_function
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
import json


REPLICAS_TO_AGGREGATE_RATIO = 1
FEATURE_COUNT = 30
HIDDEN_NODES_COUNT = 20
VALID_TRAINING_DATA_RATIO = 0.3
DELIMITER = '|'
BATCH_SIZE = 10
EPOCH = 100
WORKING_DIR = "hdfs://horton/user/webai/.yarn/"

# read from env
cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
n_pss = len(cluster_spec['ps'])  # the number of parameter servers
n_workers = len(cluster_spec['worker'])  # the number of worker nodes
job_name = os.environ["JOB_NAME"]
task_index = int(os.environ["TASK_ID"])

logging.info("job_name:%s, task_index:%d" % (job_name, task_index))

def nn_layer(input_tensor, input_dim, output_dim, scope_name, act=tf.nn.relu, act_op_name=None):
    with tf.name_scope(scope_name):
        weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1, seed=2))
        biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        activations = act(tf.matmul(input_tensor, weights) + biases, name=act_op_name)
    return activations


def model(x, y_, sample_weight):
    hidden1 = nn_layer(x, FEATURE_COUNT, HIDDEN_NODES_COUNT, "hidden_layer1")
    y = nn_layer(hidden1, HIDDEN_NODES_COUNT, 1, "output_layer", act=tf.nn.sigmoid, act_op_name="shifu_output_0")

    # count the number of updates
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss = tf.losses.mean_squared_error(predictions=y, labels=y_, weights=sample_weight)

    opt = tf.train.SyncReplicasOptimizer(
        tf.train.AdamOptimizer(0.01),
        replicas_to_aggregate=n_workers*REPLICAS_TO_AGGREGATE_RATIO,
        total_num_replicas=n_workers,
        #replica_id=task_index,  # must be worker index
        name="shifu_sync_replicas")
    train_step = opt.minimize(loss, global_step=global_step)

    return opt, train_step, loss, global_step


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

        # import data
        context = load_data(os.environ["TRAINING_DATA_PATH"])

        logging.info("Testing set size: %d" % len(context['valid_data']))
        logging.info("Training set size: %d" % len(context['train_data']))

        is_chief = (task_index == 0)  # checks if this is the chief node
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=task_index)

        # Graph
        worker_device = "/job:%s/task:%d" % (job_name, task_index)
        with tf.device(tf.train.replica_device_setter(ps_tasks=n_pss,
                                                      worker_device=worker_device,
                                                      cluster=cluster)):
            input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, FEATURE_COUNT),
                                               name="shifu_input_0")
            label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, 1))
            sample_weight_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            valid_feed = {input_placeholder: context['valid_data'],
                          label_placeholder: context['valid_target'],
                          sample_weight_placeholder : context['valid_data_sample_weight']}

            opt, train_step, loss, global_step = model(input_placeholder,
                                                       label_placeholder,
                                                       sample_weight_placeholder)

            # You can now call get_init_tokens_op() and get_chief_queue_runner().
            # Note that get_init_tokens_op() must be called before creating session
            # because it modifies the graph.
            init_token_op = opt.get_init_tokens_op()
            chief_queue_runner = opt.get_chief_queue_runner()

            # Initializing the variables
            init_op = tf.initialize_all_variables()
            logging.info("---Variables initialized---")

        # **************************************************************************************
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=WORKING_DIR,
                                 init_op=init_op,
                                 global_step=global_step,
                                 save_model_secs=10)

        # **************************************************************************************

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # **************************************************************************************
            # After the session is created by the Supervisor and before the main while loop:
            if is_chief:
                sv.start_queue_runners(sess, [chief_queue_runner])
                # Insert initial tokens to the queue.
                sess.run(init_token_op)
                time.sleep(10)
            # **************************************************************************************
            # Statistics
            net_train_t = 0
            total_batch = int(len(context["train_data"]) / BATCH_SIZE)
            # split data into batch
            input_batch = np.array_split(context["train_data"], total_batch)
            target_batch = np.array_split(context["train_target"], total_batch)
            train_sample_weight_batch = np.array_split(context["train_data_sample_weight"], total_batch)
            # Training
            for epoch in range(EPOCH):
                # Loop over all batches
                for i in range(total_batch):
                    # ======== net training time ========
                    begin_t = time.time()
                    train_feed = {input_placeholder: input_batch[i],
                                  label_placeholder: target_batch[i],
                                  sample_weight_placeholder: train_sample_weight_batch[i]}

                    _, l, gs = sess.run([train_step, loss, global_step], feed_dict=train_feed)
                    logging.info('step: ' + str(gs) + 'worker: ' + str(task_index) + " loss:" + str(l))
                    end_t = time.time()
                    net_train_t += (end_t - begin_t)
                    # ===================================
                # Calculate training accuracy
                # acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
                # print("Epoch:", '%04d' % (epoch+1), " Train Accuracy =", acc)
                print("Epoch:", '%04d' % (epoch + 1))
            print("Training Finished!")
            print("Net Training Time: ", net_train_t, "second")

        if is_chief:
            time.sleep(40)

        sv.stop()
        logging.info('Done' + str(task_index))
        sys.exit()


def load_data(data_file):
    data_file_list = data_file.split(",")
    logging.info("input data %s" % data_file_list)

    train_data = []
    train_target = []
    valid_data = []
    valid_target = []

    training_data_sample_weight = []
    valid_data_sample_weight = []

    train_pos_cnt = 0
    train_neg_cnt = 0
    valid_pos_cnt = 0
    valid_neg_cnt = 0

    # selected feature column number, it will use all feature by default
    feature_column_nums = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    sample_weight_column_num = -1
    target_column_num = 0

    file_count = 1
    line_count = 0

    for currentFile in data_file_list:
        logging.info("Now loading " + currentFile + " Progress: " + str(file_count) + "/" + str(len(data_file_list)) + ".")
        file_count += 1

        with gfile.Open(currentFile, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            while True:
                line = gf.readline()
                if len(line) == 0:
                    break

                line_count += 1
                if line_count % 10000 == 0:
                    logging.info("Total loading lines: " + str(line_count))

                columns = line.split(DELIMITER)

                if feature_column_nums == None:
                    feature_column_nums = range(0, len(columns))

                    feature_column_nums.remove(target_column_num)
                    if sample_weight_column_num >= 0:
                        feature_column_nums.remove(sample_weight_column_num)

                if random.random() >= VALID_TRAINING_DATA_RATIO:
                    # Append training data
                    train_target.append([float(columns[target_column_num])])
                    if (columns[target_column_num] == "1"):
                        train_pos_cnt += 1
                    else:
                        train_neg_cnt += 1
                    single_train_data = []
                    for feature_column_num in feature_column_nums:
                        single_train_data.append(float(columns[feature_column_num].strip('\n')))
                    train_data.append(single_train_data)

                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            logging.info("Warning: weight is below 0. example:" + line)
                            weight = 1.0
                        training_data_sample_weight.append([weight])
                    else:
                        training_data_sample_weight.append([1.0])
                else:
                    # Append validation data
                    valid_target.append([float(columns[target_column_num])])
                    if columns[target_column_num] == "1":
                        valid_pos_cnt += 1
                    else:
                        valid_neg_cnt += 1
                    single_valid_data = []
                    for feature_column_num in feature_column_nums:
                        single_valid_data.append(float(columns[feature_column_num].strip('\n')))
                    valid_data.append(single_valid_data)

                    if sample_weight_column_num >= 0 and sample_weight_column_num < len(columns):
                        weight = float(columns[sample_weight_column_num].strip('\n'))
                        if weight < 0.0:
                            logging.info("Warning: weight is below 0. example:" + line)
                            weight = 1.0
                        valid_data_sample_weight.append([weight])
                    else:
                        valid_data_sample_weight.append([1.0])

    logging.info("Total data count: " + str(line_count) + ".")
    logging.info("Train pos count: " + str(train_pos_cnt) + ", neg count: " + str(train_neg_cnt) + ".")
    logging.info("Valid pos count: " + str(valid_pos_cnt) + ", neg count: " + str(valid_neg_cnt) + ".")

    return {"train_data": train_data, "train_target": train_target,
            "valid_data": valid_data, "valid_target": valid_target,
            "train_data_sample_weight": training_data_sample_weight,
            "valid_data_sample_weight": valid_data_sample_weight,
            "feature_count": len(feature_column_nums)}


if __name__ == '__main__':
    tf.app.run()
