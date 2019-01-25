"""Synchronous Accumulated Gradients Normalization (SGAN)
Performs synchronous updates with gradients averaged
over a time window.
Author: Tommy Mulc
"""

from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os
import json
import gzip
from StringIO import StringIO
import random
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.platform import gfile

log_dir = '/logdir'

REPLICAS_TO_AGGREGATE_RATIO = 1
VALID_TRAINING_DATA_RATIO = 0.3
DELIMITER = '|'
LEARNING_RATE = 0.003
BATCH_SIZE = 10
EPOCH = 10
WORKING_DIR = "hdfs://horton/user/webai/.yarn/"

# read from env
cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
ps_hosts = cluster_spec['ps']
worker_hosts = cluster_spec['worker']
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})  # allows this node know about all other nodes
n_pss = len(cluster_spec['ps'])  # the number of parameter servers
n_workers = len(cluster_spec['worker'])  # the number of worker nodes
job_name = os.environ["JOB_NAME"]
task_index = int(os.environ["TASK_ID"])

# Configure
config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

def main():
    logging.getLogger().setLevel(logging.INFO)

    logging.info(os.environ["CLUSTER_SPEC"])
    logging.info(job_name)
    logging.info(task_index)

    # Server Setup
    if job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster,
                                 job_name="ps",
                                 task_index=task_index)
        server.join()
    else:  # it must be a worker server
        logging.info("Loading data from worker index = %d" % task_index)

        # import data
        context = load_data(os.environ["TRAINING_DATA_PATH"])
        total_batch = int(len(context["train_data"]) / BATCH_SIZE)
        # split data into batch
        input_batch = np.array_split(context["train_data"], total_batch)
        target_batch = np.array_split(context["train_target"], total_batch)
        train_sample_weight_batch = np.array_split(context["train_data_sample_weight"], total_batch)

        is_chief = (task_index == 0)  # checks if this is the chief node
        server = tf.train.Server(cluster,
                                 job_name="worker",
                                 task_index=task_index)

        # import data
        context = load_data(os.environ["TRAINING_DATA_PATH"])

        # Graph
        with tf.device("/job:worker/replica:0/task:%d" % task_index):
            input_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, FEATURE_COUNT),
                                               name="shifu_input_0")
            label_placeholder = tf.placeholder(dtype=tf.int32, shape=(None, 1))
            sample_weight_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1))

            loptimizer, loss, local_step = model(input_placeholder, label_placeholder, sample_weight_placeholder)

            # SDAG (simplest case since all batches are the same)
            update_window = 5  # T: communication window
            grad_list = None  # the array to store the gradients through the communication window
            for t in range(update_window):
                if t != 0:
                    # compute gradients only if the local opt was run
                    with tf.control_dependencies([opt_local]):
                        grads, varss = zip(*loptimizer.compute_gradients(
                            loss, var_list=tf.local_variables()))
                else:
                    grads, varss = zip(*loptimizer.compute_gradients(
                        loss, var_list=tf.local_variables()))

                # add gradients to the list
                if grad_list:
                    for i in range(len(grads)):
                        grad_list[i].append(grads[i])
                else:
                    # Init grad list
                    grad_list = []
                    for grad in grads:
                        grad_list.append([grad])

                # update local parameters
                opt_local = loptimizer.apply_gradients(zip(grads, varss),
                                                       global_step=local_step)

            # averages updates before applying globally
            grad_tuple = []
            for grad in grad_list:
                grad_tuple.append(tf.reduce_mean(grad, axis=0))

            grads = tuple(grad_tuple)

            # add these variables created by local optimizer to local collection
            lopt_vars = add_global_variables_to_local_collection()

            # delete the variables from the global collection
            clear_global_collection()

        with tf.device(tf.train.replica_device_setter(ps_tasks=n_pss,
                                                      worker_device="/job:%s/task:%d" % (job_name, task_index))):

            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            # create global variables and/or references
            local_to_global, global_to_local = create_global_variables(lopt_vars)

            optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            optimizer1 = tf.train.SyncReplicasOptimizer(optimizer,
                                                        replicas_to_aggregate=int(n_workers * REPLICAS_TO_AGGREGATE_RATIO),
                                                        total_num_replicas=n_workers)

            # apply the gradients to variables on ps
            opt = optimizer1.apply_gradients(zip(grads, [local_to_global[v] for v in varss]), global_step=global_step)

            with tf.control_dependencies([opt]):
                assign_locals = assign_global_to_local(global_to_local)

            # Grab global state before training so all workers have same initialization
            grab_global_init = assign_global_to_local(global_to_local)

            # Assigns local values to global ones for chief to execute
            assign_global = assign_local_to_global(local_to_global)

            # Initialized global step tokens
            init_tokens_op = optimizer1.get_init_tokens_op()

            # Init ops
            # gets step token
            local_init = optimizer1.local_step_init_op
            if is_chief:
                # fills token queue and gets token
                local_init = optimizer1.chief_init_op

            # indicates if variables are initialized
            ready_for_local_init = optimizer1.ready_for_local_init_op

            with tf.control_dependencies([local_init]):
                init_local = tf.variables_initializer(tf.local_variables()
                                                      + tf.get_collection('local_non_trainable'))  # for local variables

            init = tf.global_variables_initializer()  # must come after other init ops

        # Session
        sync_replicas_hook = optimizer1.make_session_run_hook(is_chief)
        stop_hook = tf.train.StopAtStepHook(last_step=EPOCH * total_batch)  # epoch * total_batch) # step means every step to update variable
        chief_hooks = [sync_replicas_hook, stop_hook]
        scaff = tf.train.Scaffold(init_op=init,
                                  local_init_op=init_local,
                                  ready_for_local_init_op=ready_for_local_init)

        # Monitored Training Session
        sess = tf.train.MonitoredTrainingSession(master=server.target,
                                                 is_chief=is_chief,
                                                 config=config,
                                                 scaffold=scaff,
                                                 hooks=chief_hooks,
                                                 stop_grace_period_secs=10)

        if is_chief:
            sess.run(assign_global)  # Assigns chief's initial values to ps
            time.sleep(40)  # grace period to wait on other workers before starting training

        # Train until hook stops session
        print('Starting training on worker %d' % task_index)
        sess.run(grab_global_init)

        # Train until hook stops session
        print('Starting training on worker %d' % task_index)
        cur_epoch = 1
        while not sess.should_stop():
            sum_train_error = 0.0
            for i in range(total_batch):
                _, _, l, gs, ls = sess.run([opt, assign_locals, loss, global_step, local_step],
                                           feed_dict={
                                               input_placeholder: input_batch[i],
                                               label_placeholder: target_batch[i],
                                               sample_weight_placeholder: train_sample_weight_batch[i],
                                           })
                sum_train_error += l
            # _,r,gs=sess.run([opt,c,global_step])
            print("Epoch ", cur_epoch, sum_train_error, gs, task_index)
            if is_chief: time.sleep(1)
            time.sleep(1)
            cur_epoch += 1
        print('Done', task_index)

        time.sleep(10)  # grace period to wait before closing session
        sess.close()
        print('Session from worker %d closed cleanly' % task_index)


def nn_layer(input_tensor, input_dim, output_dim, scope_name, act=tf.nn.relu, act_op_name=None):
    with tf.name_scope(scope_name):
        weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1, seed=2), collections=[tf.GraphKeys.LOCAL_VARIABLES])
        biases = tf.Variable(tf.constant(0.1, shape=[output_dim]), collections=[tf.GraphKeys.LOCAL_VARIABLES])
        activations = act(tf.matmul(input_tensor, weights) + biases, name=act_op_name)
    return activations


def model(x, y_, sample_weight):
    hidden1 = nn_layer(x, FEATURE_COUNT, HIDDEN_NODES_COUNT, "hidden_layer1")
    y = nn_layer(hidden1, HIDDEN_NODES_COUNT, 1, "output_layer", act=tf.nn.sigmoid, act_op_name="shifu_output_0")

    # count the number of updates
    local_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                             name='local_step', collections=['local_non_trainable'])

    loss = tf.losses.mean_squared_error(predictions=y, labels=y_, weights=sample_weight)

    loptimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

    return loptimizer, loss, local_step


def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
    print("saving model......")
    remove_path(export_dir)
    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature_def_utils.predict_signature_def(inputs, outputs)
    }
    b = builder.SavedModelBuilder(export_dir)
    b.add_meta_graph_and_variables(
        session,
        tags=[tag_constants.SERVING],
        signature_def_map=signature_def_map,
        assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
        legacy_init_op=legacy_init_op,
        clear_devices=True)
    b.save()
    print("saved model to ", export_dir)
    export_generic_config(export_dir=export_dir)


def remove_path(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) and os.path.exists(path):
        os.remove(path)
        return
    files = os.listdir(path)
    for f in files:
        remove_path(path + "/" + f)
    os.removedirs(path)


def export_generic_config(export_dir):
    config_json_str = ""
    config_json_str += "{\n"
    config_json_str += "    \"inputnames\": [\n"
    config_json_str += "        \"shifu_input_0\"\n"
    config_json_str += "      ],\n"
    config_json_str += "    \"properties\": {\n"
    config_json_str += "         \"algorithm\": \"tensorflow\",\n"
    config_json_str += "         \"tags\": [\"serve\"],\n"
    config_json_str += "         \"outputnames\": \"shifu_output_0\",\n"
    config_json_str += "         \"normtype\": \"ZSCALE\"\n"
    config_json_str += "      }\n"
    config_json_str += "}"
    f = file(export_dir + "/" + "GenericModelConfig.json", mode="w+")
    f.write(config_json_str)


def load_data(dataFileList):
    dataFileList = dataFileList.split(",")
    print("input data %s" % dataFileList)

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

    feature_column_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                           26, 27, 28, 29, 30]  # selected feature column number, it will use all feature by default
    sample_weight_column_num = -1
    target_column_num = 0

    file_count = 1
    line_count = 0

    for currentFile in dataFileList:
        print("Now loading " + currentFile + " Progress: " + str(file_count) + "/" + str(len(dataFileList)) + ".")
        file_count += 1

        with gfile.Open(currentFile, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            while True:
                line = gf.readline()
                if len(line) == 0:
                    break

                line_count += 1
                if line_count % 10000 == 0:
                    tprint("Total loading lines: " + str(line_count))

                columns = line.split(DELIMITER)

                if feature_column_nums == None:
                    feature_column_nums = range(0, len(columns))

                    feature_column_nums.remove(target_column_num)
                    if sample_weight_column_num >= 0:
                        feature_column_nums.remove(sample_weight_column_num)

                feature_cnt = len(feature_column_nums)

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
                            print("Warning: weight is below 0. example:" + line)
                            weight = 1.0
                        training_data_sample_weight.append([weight])
                    else:
                        training_data_sample_weight.append([1.0])
                else:
                    # Append validation data
                    valid_target.append([float(columns[target_column_num])])
                    if (columns[target_column_num] == "1"):
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
                            print("Warning: weight is below 0. example:" + line)
                            weight = 1.0
                        valid_data_sample_weight.append([weight])
                    else:
                        valid_data_sample_weight.append([1.0])

    print("Total data count: " + str(line_count) + ".")
    print("Train pos count: " + str(train_pos_cnt) + ", neg count: " + str(train_neg_cnt) + ".")
    print("Valid pos count: " + str(valid_pos_cnt) + ", neg count: " + str(valid_neg_cnt) + ".")

    return {"train_data": train_data, "train_target": train_target, \
            "valid_data": valid_data, "valid_target": valid_target, \
            "train_data_sample_weight": training_data_sample_weight, \
            "valid_data_sample_weight": valid_data_sample_weight, \
            "feature_count": len(feature_column_nums)}


def assign_global_to_local(global_to_local):
    """Assigns global variable value to local variables.
	global_to_local : dictionary with corresponding local variable for global key
	"""
    r = []
    for v in global_to_local.keys():
        r.append(tf.assign(global_to_local[v], v))
    with tf.control_dependencies(r):
        a = tf.no_op()
    return a


def assign_local_to_global(local_to_global):
    """Assigns global variable value to local variables.
	local_to_global : dictionary with corresponding global variable for local key
	"""
    r = []
    for v in local_to_global.keys():
        r.append(tf.assign(local_to_global[v], v))
    with tf.control_dependencies(r):
        a = tf.no_op()
    return a


def get_variable_by_name(name):
    """Returns the variable of given name
	name : the name of the global variable
	"""
    return [v for v in tf.get_collection('variables') if v.name == name][0]


def get_global_variable_by_name(name):
    """Returns the global variable of given name.
	name : the name of the global variable
	"""
    # return [v for v in tf.variables() if v.name == name][0]
    return [v for v in tf.global_variables() if v.name == name][0]


def create_global_variables(local_optimizer_vars=[]):
    """Creates global variables for local variables on the graph.
	Skips variables local variables that are created for
	local optimization.
	Returns dictionarys for local-to-global and global-to-local
	variable mappings.
	"""
    local_to_global = {}
    global_to_local = {}
    with tf.device('/job:ps/task:0'):
        for v in tf.local_variables():
            if v not in local_optimizer_vars:
                v_g = tf.get_variable('g/' + v.op.name,
                                      shape=v.shape,
                                      dtype=v.dtype,
                                      trainable=True,
                                      collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                   tf.GraphKeys.TRAINABLE_VARIABLES])
                local_to_global[v] = v_g
                global_to_local[v_g] = v
    return local_to_global, global_to_local


def add_global_variables_to_local_collection():
    """Adds all variables from the global collection
	to the local collection.
	Returns the list of variables added.
	"""
    r = []
    for var in tf.get_default_graph()._collections[tf.GraphKeys.GLOBAL_VARIABLES]:
        tf.add_to_collection(tf.GraphKeys.LOCAL_VARIABLES, var)
        r.append(var)
    return r


def clear_global_collection():
    """Removes all variables from global collection."""
    g = tf.get_default_graph()
    for _ in range(len(g._collections[tf.GraphKeys.GLOBAL_VARIABLES])):
        del g._collections[tf.GraphKeys.GLOBAL_VARIABLES][0]


if __name__ == '__main__':
    main()
