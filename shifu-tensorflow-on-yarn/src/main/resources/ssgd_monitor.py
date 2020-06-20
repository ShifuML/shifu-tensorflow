"""Synchronous SGD
"""

# from __future__ import print_function
import os
import tensorflow as tf
import time
import sys
import logging
import gzip
from io import BytesIO
import random
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
import json
import socket
from threading import Thread
import tensorboard.main as tb_main

tf.compat.v1.disable_eager_execution()

HIDDEN_NODES_COUNT = 20
VALID_TRAINING_DATA_RATIO = 0.1

BUILD_MODEL_BY_CONF_ENABLE = True
REPLICAS_TO_AGGREGATE_RATIO = 1

DELIMITER = '|'
BATCH_SIZE = 100
TB_PORT_ENV_VAR = 'TB_PORT'

# read from env
cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
n_pss = len(cluster_spec['ps'])  # the number of parameter servers
n_workers = int(os.environ["WORKER_CNT"])  # the number of worker nodes
job_name = os.environ["JOB_NAME"]
task_index = int(os.environ["TASK_ID"])
socket_server_port = int(os.environ["SOCKET_SERVER_PORT"])  # The port of local java socket server listening, to sync worker training intermediate information with master
total_training_data_number = int(os.environ["TOTAL_TRAINING_DATA_NUMBER"]) # total data
feature_column_nums = [int(s) for s in str(os.environ["SELECTED_COLUMN_NUMS"]).split(' ')]  # selected column numbers
FEATURE_COUNT = len(feature_column_nums)

sample_weight_column_num = int(os.environ["WEIGHT_COLUMN_NUM"])  # weight column number, default is -1
target_column_num = int(os.environ["TARGET_COLUMN_NUM"])  # target column number, default is -1

tmp_model_path = os.environ["TMP_MODEL_PATH"]
final_model_path = os.environ["FINAL_MODEL_PATH"]

# This client is used for sync worker training intermediate information with master
socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_client.connect(("127.0.0.1", socket_server_port))


def nn_layer(input_tensor, input_dim, output_dim, act=tf.nn.tanh, act_op_name=None):
    l2_reg = tf.keras.regularizers.l2(l=0.1)
    weights = tf.compat.v1.get_variable(name="weight_"+str(act_op_name),
                              shape=[input_dim, output_dim],
                              regularizer=l2_reg,
                              initializer=tf.initializers.GlorotUniform())
    biases = tf.compat.v1.get_variable(name="biases_"+str(act_op_name),
                             shape=[output_dim],
                             regularizer=l2_reg,
                             initializer=tf.initializers.GlorotUniform())

    activations = act(tf.compat.v1.matmul(input_tensor, weights) + biases, name=act_op_name)
    return activations


def get_activation_fun(name):
    if name is None:
        return tf.compat.v1.nn.leaky_relu
    name = name.lower()

    if 'sigmoid' == name:
        return tf.compat.v1.nn.sigmoid
    elif 'tanh' == name:
        return tf.compat.v1.nn.tanh
    elif 'relu' == name:
        return tf.compat.v1.nn.relu
    elif 'leakyrelu' == name:
        return tf.compat.v1.nn.leaky_relu
    else:
        return tf.compat.v1.nn.leaky_relu


def generate_from_modelconf(x, model_conf):
    train_params = model_conf['train']['params']
    num_hidden_layer = int(train_params['NumHiddenLayers'])
    num_hidden_nodes = [int(s) for s in train_params['NumHiddenNodes']]
    activation_func = [get_activation_fun(s) for s in train_params['ActivationFunc']]

    global FEATURE_COUNT
    # first layer
    previous_layer = nn_layer(x, FEATURE_COUNT, num_hidden_nodes[0],
                     act=activation_func[0], act_op_name="hidden_layer" + str(0))

    for i in range(1, num_hidden_layer):
        layer = nn_layer(previous_layer, num_hidden_nodes[i-1], num_hidden_nodes[i],
                     act=activation_func[i], act_op_name="hidden_layer" + str(i))
        previous_layer = layer

    return previous_layer, num_hidden_nodes[num_hidden_layer-1]


def model(x, y_, sample_weight, model_conf):
    logging.info("worker_num:%d" % n_workers)
    logging.info("total_training_data_number:%d" % total_training_data_number)

    if BUILD_MODEL_BY_CONF_ENABLE and model_conf is not None:
        output_digits, output_nodes = generate_from_modelconf(x, model_conf)
    else:
        output_digits = nn_layer(x, FEATURE_COUNT, HIDDEN_NODES_COUNT, act_op_name="hidden_layer1")
        output_nodes = HIDDEN_NODES_COUNT

    logging.info("output_nodes : " + str(output_nodes))
    y = nn_layer(output_digits, output_nodes, 1, act=tf.compat.v1.nn.sigmoid, act_op_name="shifu_output_0")

    # count the number of updates
    global_step = tf.compat.v1.get_variable('global_step', [],
                                  initializer=tf.compat.v1.constant_initializer(0),
                                  trainable=False,
                                  dtype=tf.compat.v1.int32)

    loss = tf.compat.v1.losses.mean_squared_error(predictions=y, labels=y_, weights=sample_weight)

    # we suppose every worker has same batch_size
    if model_conf is not None:
        learning_rate = model_conf['train']['params']['LearningRate']
    else:
        learning_rate = 0.003
    opt = tf.compat.v1.train.SyncReplicasOptimizer(
        #tf.train.GradientDescentOptimizer(learning_rate),
        tf.compat.v1.train.AdadeltaOptimizer(learning_rate=learning_rate),
        replicas_to_aggregate=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE * REPLICAS_TO_AGGREGATE_RATIO),
        total_num_replicas=int(total_training_data_number * (1-VALID_TRAINING_DATA_RATIO) / BATCH_SIZE),
        name="shifu_sync_replicas")
    train_step = opt.minimize(loss, global_step=global_step)

    return opt, train_step, loss, global_step, y


def main(_):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    logging.info("job_name:%s, task_index:%d" % (job_name, task_index))

    ps_hosts = cluster_spec['ps']
    worker_hosts = cluster_spec['worker']
    cluster = tf.compat.v1.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # allows this node know about all other nodes
    if job_name == 'ps':  # checks if parameter server
        server = tf.compat.v1.train.Server(cluster,
                                 job_name="ps",
                                 task_index=task_index)
        server.join()
    else:  # it must be a worker server
        is_chief = (task_index == 0)  # checks if this is the chief node
        server = tf.compat.v1.train.Server(cluster,
                                 job_name="worker",
                                 task_index=task_index)

        logging.info("Loading data from worker index = %d" % task_index)

        if "TRAINING_DATA_PATH" in os.environ:
            logging.info("This is a normal worker..")
            training_data_path = os.environ["TRAINING_DATA_PATH"]
        else:
            logging.info("This is a backup worker")
            # watching certain file in hdfs which contains its training data

        # Read model structure info from ModelConfig
        with open('./ModelConfig.json') as f:
            model_conf = json.load(f)
            logging.info("model" + str(model_conf))
            EPOCH = int(model_conf['train']['numTrainEpochs'])
            global VALID_TRAINING_DATA_RATIO
            VALID_TRAINING_DATA_RATIO = model_conf['train']['validSetRate']

        # import data
        context = load_data(training_data_path)

        # split data into batch
        total_batch = int(len(context["train_data"]) / BATCH_SIZE)
        x_batch = np.array_split(context["train_data"], total_batch)
        y_batch = np.array_split(context["train_target"], total_batch)
        sample_w_batch = np.array_split(context["train_data_sample_weight"], total_batch)

        logging.info("Testing set size: %d" % len(context['valid_data']))
        logging.info("Training set size: %d" % len(context['train_data']))

        valid_x = np.asarray(context["valid_data"])
        valid_y = np.asarray(context["valid_target"])
        valid_sample_w = np.asarray(context["valid_data_sample_weight"])

        # Graph
        worker_device = "/job:%s/task:%d" % (job_name, task_index)

        with tf.compat.v1.device(tf.compat.v1.train.replica_device_setter(
                                                      cluster=cluster,
                                                      worker_device=worker_device
                                                      )):
            input_placeholder = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=(None, FEATURE_COUNT),
                                               name="shifu_input_0")
            label_placeholder = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=(None, 1))
            sample_weight_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1))

            opt, train_step, loss, global_step, y = model(input_placeholder,
                                                          label_placeholder,
                                                          sample_weight_placeholder,
                                                          model_conf)

            # init ops
            init_tokens_op = opt.get_init_tokens_op()
            # initialize local step
            local_init = opt.local_step_init_op
            if is_chief:
                # initializes token queue
                local_init = opt.chief_init_op

            # checks if global vars are init
            ready_for_local_init = opt.ready_for_local_init_op

            # Initializing the variables
            init_op = tf.compat.v1.initialize_all_variables()
            logging.info("---Variables initialized---")

        # **************************************************************************************
        # Session
        sync_replicas_hook = opt.make_session_run_hook(is_chief)
        stop_hook = tf.compat.v1.train.StopAtStepHook(num_steps=EPOCH)
        chief_hooks = [sync_replicas_hook, stop_hook]
        scaff = tf.compat.v1.train.Scaffold(init_op=init_op,
                                  local_init_op=local_init,
                                  ready_for_local_init_op=ready_for_local_init)
        # Configure
        if "IS_BACKUP" in os.environ:
            config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True,
                                    device_filters=['/job:ps', '/job:worker/task:0',
                                                    '/job:worker/task:%d' % task_index])
        else:
            config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                    allow_soft_placement=True)

        # Create a "supervisor", which oversees the training process.
        sess = tf.compat.v1.train.MonitoredTrainingSession(master=server.target,
                                                 is_chief=is_chief,
                                                 config=config,
                                                 scaffold=scaff,
                                                 hooks=chief_hooks,
                                                 stop_grace_period_secs=10,
                                                 checkpoint_dir=tmp_model_path)

        if is_chief:
            sess.run(init_tokens_op)
            #start_tensorboard(tmp_model_path)
            logging.info("chief start waiting 40 sec")
            time.sleep(40)  # grace period to wait on other workers before starting training
            logging.info("chief finish waiting 40 sec")

        # Train until hook stops session
        logging.info('Starting training on worker %d' % task_index)
        while not sess.should_stop():
            try:
                start = time.time()
                for i in range(total_batch):
                    train_feed = {input_placeholder: x_batch[i],
                                  label_placeholder: y_batch[i],
                                  sample_weight_placeholder: sample_w_batch[i]}

                    _, l, gs = sess.run([train_step, loss, global_step], feed_dict=train_feed)
                training_time = time.time() - start

                time.sleep(5)

                valid_loss, gs = sess.run([loss, global_step], feed_dict={input_placeholder: valid_x,
                                                                          label_placeholder: valid_y,
                                                                          sample_weight_placeholder: valid_sample_w}
                                          )
                logging.info('Step: ' + str(gs) + ' worker: ' + str(task_index) + " training loss:" + str(l) + " valid loss:" + str(valid_loss))

                # Send intermediate result to master
                message = "worker_index:{},time:{},current_epoch:{},training_loss:{},valid_loss:{}\n".format(
                    str(task_index), str(training_time), str(gs), str(l), str(valid_loss))
                if sys.version_info < (3, 0):
                    socket_client.send(bytes(message))
                else:
                    socket_client.send(bytes(message), 'utf8')

            except RuntimeError as re:
                if 'Run called even after should_stop requested.' == re.args[0]:
                    logging.info('About to execute sync_clean_up_op!')
                else:
                    raise

        logging.info('Done' + str(task_index))

        # We just need to make sure chief worker exit with success status is enough
        if is_chief:
            tf.compat.v1.reset_default_graph()

            # add placeholders for input images (and optional labels)
            x = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=(None, FEATURE_COUNT),
                               name="shifu_input_0")
            with tf.compat.v1.get_default_graph().as_default():
                if BUILD_MODEL_BY_CONF_ENABLE and model_conf is not None:
                    output_digits, output_nodes = generate_from_modelconf(x, model_conf)
                else:
                    output_digits = nn_layer(x, FEATURE_COUNT, HIDDEN_NODES_COUNT, act_op_name="hidden_layer1")
                    output_nodes = HIDDEN_NODES_COUNT

                logging.info("output_nodes : " + str(output_nodes))
                prediction = nn_layer(output_digits, output_nodes, 1, act=tf.compat.v1.nn.sigmoid,
                                      act_op_name="shifu_output_0")

            # restore from last checkpoint
            saver = tf.compat.v1.train.Saver()
            with tf.compat.v1.Session() as sess:
                ckpt = tf.compat.v1.train.get_checkpoint_state(tmp_model_path)
                logging.info("ckpt: {}".format(ckpt))
                assert ckpt, "Invalid model checkpoint path: {}".format(tmp_model_path)
                saver.restore(sess, ckpt.model_checkpoint_path)

                logging.info("Exporting saved_model to: {}".format(final_model_path))

                # exported signatures defined in code
                simple_save(session=sess, export_dir=final_model_path,
                            inputs={
                                "shifu_input_0": x
                            },
                            outputs={
                                "shifu_output_0": prediction
                            })
                logging.info("Exported saved_model")

            time.sleep(40) # grace period to wait before closing session

        #sess.close()
        logging.info('Session from worker %d closed cleanly' % task_index)
        sys.exit()


def load_data(data_file):
    data_file_list = data_file.split(",")
    global feature_column_nums

    logging.info("input data %s" % data_file_list)
    logging.info("SELECTED_COLUMN_NUMS" + str(feature_column_nums))

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

    file_count = 1
    line_count = 0

    for currentFile in data_file_list:
        logging.info(
            "Now loading " + currentFile + " Progress: " + str(file_count) + "/" + str(len(data_file_list)) + ".")
        file_count += 1

        with gfile.Open(currentFile, 'rb') as f:
            gf = gzip.GzipFile(fileobj=BytesIO(f.read()))
            while True:
                line = gf.readline().decode()
                if len(line) == 0:
                    break

                line_count += 1
                if line_count % 10000 == 0:
                    logging.info("Total loading lines: " + str(line_count))

                columns = line.split(DELIMITER)

                if feature_column_nums is None:
                    feature_column_nums = list(range(0, len(columns)))

                    feature_column_nums.remove(target_column_num)
                    if sample_weight_column_num >= 0:
                        feature_column_nums.remove(sample_weight_column_num)

                if random.random() >= VALID_TRAINING_DATA_RATIO:
                    # Append training data
                    train_target.append([float(columns[target_column_num])])
                    if columns[target_column_num] == "1":
                        train_pos_cnt += 1
                    else:
                        train_neg_cnt += 1
                    single_train_data = []
                    for feature_column_num in feature_column_nums:
                        try:
                            single_train_data.append(float(columns[feature_column_num].strip('\n')))
                        except:
                            logging.info("Could not convert " + str(columns[feature_column_num].strip('\n') + " to float"))
                            logging.info("feature_column_num: " + str(feature_column_num))
                    train_data.append(single_train_data)

                    if 0 <= sample_weight_column_num < len(columns):
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
                        try:
                            single_valid_data.append(float(columns[feature_column_num].strip('\n')))
                        except:
                            logging.info("Could not convert " + str(columns[feature_column_num].strip('\n') + " to float"))
                            logging.info("feature_column_num: " + str(feature_column_num))

                    valid_data.append(single_valid_data)

                    if 0 <= sample_weight_column_num < len(columns):
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


def simple_save(session, export_dir, inputs, outputs, legacy_init_op=None):
    if tf.compat.v1.gfile.Exists(export_dir):
        tf.compat.v1.gfile.DeleteRecursively(export_dir)
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
    export_generic_config(export_dir=export_dir)


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
    f = tf.compat.v1.gfile.GFile(export_dir + "/GenericModelConfig.json", mode="w+")
    f.write(config_json_str)


def start_tensorboard(checkpoint_dir):
    tf.compat.v1.flags.FLAGS.logdir = checkpoint_dir
    if TB_PORT_ENV_VAR in os.environ:
        tf.compat.v1.flags.FLAGS.port = os.environ['TB_PORT']

    tb_thread = Thread(target=tb_main.run_main)
    tb_thread.daemon = True

    logging.info("Starting TensorBoard with --logdir=" + checkpoint_dir + " in daemon thread...")
    tb_thread.start()


if __name__ == '__main__':
    tf.compat.v1.app.run()
