/*
 * Copyright [2013-2018] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.yarn.container;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.ServerSocket;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooDefs.Ids;

import ml.shifu.guagua.coordinator.zk.GuaguaZooKeeper;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.Constants;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;
import ml.shifu.shifu.core.yarn.util.HdfsUtils;
import ml.shifu.shifu.util.HDFSUtils;

/**
 * Content that we want to run in the containers. TaskExecutor will register itself with AM and fetch cluster spec from
 * AM. After the cluster spec is collected, TaskExecutor will set up local environment and start the worker task.
 */
public class TensorflowTaskExecutor implements Watcher {
    private static final Log LOG = LogFactory.getLog(TensorflowTaskExecutor.class);

    /** Use for sync port with master to build tensorflow cluster **/
    private GuaguaZooKeeper zookeeper;
    private String containerId;
    private boolean isBackup;
    private String taskIndex;
    private String jobName;

    /** Use for wait all workers registering on Master **/
    final CountDownLatch latch = new CountDownLatch(1);
    public final CountDownLatch backupStartingLatch = new CountDownLatch(1);

    private String tensorflowCluster;
    /** Use for reserve port for tensorflow cluster **/
    private ServerSocket tensorflowSocket;
    private String tensorflowPort;
    /** Use for tensor board **/
    private int tbPort;
    /** Use for communicate between python program and java **/
    private SocketServer socketServer;

    private Configuration globalConf = new Configuration();

    private Map<String, String> shellEnv = new HashMap<String, String>();

    private String pythonScript;
    private String pythonShell;

    /** Process of executing back-up python script **/
    private Process backupProcess = null;

    public TensorflowTaskExecutor() {
        globalConf.addResource(new Path(Constants.GLOBAL_FINAL_XML));
    }

    public void registeryToCluster() throws KeeperException, InterruptedException, IOException {
        tensorflowPort = getTensorflowPort();

        if(StringUtils.isBlank(tensorflowPort)) {
            throw new RuntimeException("Given port on container is blank!");
        }

        // Keep watching final cluster. if master write final cluster, it means workers could continue training script
        zookeeper.exists(Constants.TENSORFLOW_FINAL_CLUSTER, true);
        zookeeper.exists(Constants.getTrainingDataZookeeperPath(containerId), true);

        zookeeper.createOrSetExt(Constants.TENSORFLOW_CLUSTER_ROOT_PATH + containerId,
                (CommonUtils.getCurrentHostIP() + ":" + tensorflowPort).getBytes(Charset.forName("UTF-8")),
                Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT, true, -1);

        latch.await();

        this.shellEnv.put("CLUSTER_SPEC", this.tensorflowCluster);// json{'ps':[1.1.1.1:123,1.1.1.2:123],'worker':[1.1.1.1:123]}
    }

    // callback method would be trigger if master collect all workers ip and port
    public void process(WatchedEvent event) {
        LOG.info("event path:" + event.getPath());
        if(Watcher.Event.EventType.NodeCreated == event.getType()
                && Constants.TENSORFLOW_FINAL_CLUSTER.equalsIgnoreCase(event.getPath())) {
            try {
                tensorflowCluster = new String(zookeeper.getData(Constants.TENSORFLOW_FINAL_CLUSTER, false, null));

                // reset taskid due to it may changed because current task could be backup task and replaced
                // slow-register task
                if(this.isBackup) {
                    String[] fields = tensorflowCluster.split("worker")[1].split(",");
                    for(int i = 0; i < fields.length; i++) {
                        if(fields[i].contains(CommonUtils.getCurrentHostIP() + ":" + tensorflowPort)) {
                            if(Integer.valueOf(taskIndex) != i) {
                                LOG.info("change taskid from " + taskIndex + " to " + i);
                                shellEnv.put("TASK_ID", Integer.toString(i));
                                this.isBackup = false;
                            }
                            break;
                        }
                    }
                    // we use order in cluster as taskID for ps as well, because we will igore a few ps with slow
                    // starter
                } else if(Constants.PS_JOB_NAME.equalsIgnoreCase(this.jobName)) {
                    String[] fields = tensorflowCluster.split("worker")[0].split(",");
                    for(int i = 0; i < fields.length; i++) {
                        if(fields[i].contains(CommonUtils.getCurrentHostIP() + ":" + tensorflowPort)) {
                            if(Integer.valueOf(taskIndex) != i) {
                                LOG.info("change taskid from " + taskIndex + " to " + i);
                                shellEnv.put("TASK_ID", Integer.toString(i));
                            }
                            break;
                        }
                    }
                }

                LOG.info("Cluster:" + tensorflowCluster);
                latch.countDown();
            } catch (Exception e) {
                LOG.error("Error when getting final cluster from zookeeper", e);
                throw new RuntimeException(e);
            }
        } else if(Watcher.Event.EventType.NodeCreated == event.getType()
                && Constants.getTrainingDataZookeeperPath(containerId).equalsIgnoreCase(event.getPath())) {
            LOG.info("Weak up back host!!");

            try {
                String trainingDataPath = new String(
                        zookeeper.getData(Constants.getTrainingDataZookeeperPath(containerId), false, null));

                LOG.info("TRAINING_DATA_PATH: " + trainingDataPath);
                shellEnv.put("TRAINING_DATA_PATH", trainingDataPath);

                if(backupProcess != null) {
                    CommonUtils.killProcessByPort(tensorflowPort);
                    backupProcess.destroy();
                    LOG.info("Killed backup waiting program... " + backupProcess.isAlive());
                }

                backupStartingLatch.countDown();
            } catch (Exception e) {
                LOG.error("Error when getting backup training data path from zookeeper", e);
                throw new RuntimeException(e);
            }
        }
    }

    public String getTensorflowPort() throws IOException {
        tensorflowSocket = new ServerSocket(CommonUtils.getValidTensorflowPort());

        return Integer.toString(this.tensorflowSocket.getLocalPort());
    }

    public static CommandLine initOpts(String[] args) throws ParseException {
        Options opts = new Options();

        opts.addOption("job_name", true, "");
        opts.addOption("task_id", true, "");
        opts.addOption("container_id", true, "");
        opts.addOption("training_data_path", true, "");
        opts.addOption("zookeeper_server", true, "");
        opts.addOption("is_backup", true, "");

        return new GnuParser().parse(opts, args);
    }

    public void init(CommandLine cliParser) throws ParseException, IOException, InterruptedException {
        // Use in train.py
        taskIndex = cliParser.getOptionValue("task_id");
        jobName = cliParser.getOptionValue("job_name");
        shellEnv.put("JOB_NAME", jobName); // worker or ps
        shellEnv.put("TASK_ID", taskIndex);
        shellEnv.put("TRAINING_DATA_PATH", cliParser.getOptionValue("training_data_path")); // /path/a,/path/b
        // total training data count
        shellEnv.put("TOTAL_TRAINING_DATA_NUMBER", globalConf.get(GlobalConfigurationKeys.TOTAL_TRAINING_DATA_NUM));
        shellEnv.put("WEIGHT_COLUMN_NUM", globalConf.get(GlobalConfigurationKeys.WEIGHT_COLUMN_NUM,
                GlobalConfigurationKeys.DEFAULT_WEIGHT_COLUMN_NUM)); // default is -1.
        shellEnv.put("TARGET_COLUMN_NUM", globalConf.get(GlobalConfigurationKeys.TARGET_COLUMN_NUM,
                GlobalConfigurationKeys.DEFAULT_TARGET_COLUMN_NUM));
        if(StringUtils.isNotBlank(globalConf.get(GlobalConfigurationKeys.SELECTED_COLUMN_NUMS))) {
            shellEnv.put("SELECTED_COLUMN_NUMS", globalConf.get(GlobalConfigurationKeys.SELECTED_COLUMN_NUMS,
                    GlobalConfigurationKeys.DEFAULT_SELECTED_COLUMN_NUMS)); // default is -1
        } else {
            shellEnv.put("SELECTED_NUMERIC_COLUMN_NUMS",
                    globalConf.get(GlobalConfigurationKeys.SELECTED_NUMERIC_COLUMN_NUMS,
                            GlobalConfigurationKeys.DEFAULT_SELECTED_COLUMN_NUMS)); // default is -1
            shellEnv.put("SELECTED_CATEGORY_COLUMN_NUMS",
                    globalConf.get(GlobalConfigurationKeys.SELECTED_CATEGORY_COLUMN_NUMS,
                            GlobalConfigurationKeys.DEFAULT_SELECTED_COLUMN_NUMS)); // default is -1
        }

        shellEnv.put("TMP_MODEL_PATH", globalConf.get(GlobalConfigurationKeys.TMP_MODEL_PATH));
        shellEnv.put("FINAL_MODEL_PATH", globalConf.get(GlobalConfigurationKeys.FINAL_MODEL_PATH));

        containerId = cliParser.getOptionValue("container_id").trim();
        isBackup = Boolean.valueOf(cliParser.getOptionValue("is_backup"));

        String zookeeperServer = cliParser.getOptionValue("zookeeper_server");
        zookeeper = new GuaguaZooKeeper(zookeeperServer, 3000000, 5, 1000, this);

        // start socket server so that python could connect to this server to sending message
        socketServer = new SocketServer(zookeeper, containerId);
        socketServer.start();
        shellEnv.put("SOCKET_SERVER_PORT", Integer.toString(socketServer.getServerPort()));
    }

    public void prepare() throws IOException {
        if(new File(Constants.PYTHON_VENV_ZIP).exists()) {
            LOG.info("Unpacking Python virtual environment.. ");
            CommonUtils.unzipArchive(Constants.PYTHON_VENV_ZIP, ".");
        } else {
            LOG.info("No virtual environment uploaded.");
        }

        if(new File(Constants.GLIBC_VENV_ZIP).exists()) {
            LOG.info("Unpacking Python virtual environment.. ");
            CommonUtils.unzipArchive(Constants.GLIBC_VENV_ZIP, ".");
        } else {
            LOG.info("No virtual environment uploaded.");
        }

        String pythonBinaryPath = globalConf.get(GlobalConfigurationKeys.PYTHON_BINARY_PATH);
        String glibcBinaryPath = globalConf.get(GlobalConfigurationKeys.GLIBC_BINARY_PATH);
        // Use in bash
        shellEnv.put("GLIBC_HOME", "." + glibcBinaryPath);
        shellEnv.put("PYTHON_HOME", "." + pythonBinaryPath);

        // Copy backup script so that we could execute
        // We put backup script into yarn.jar due to user no need to touch it
        InputStream inputStream = null;
        try {
            inputStream = this.getClass().getResourceAsStream(Constants.BACKUP_SCRIPT);
            Files.copy(inputStream, Paths.get("." + Constants.BACKUP_SCRIPT), StandardCopyOption.REPLACE_EXISTING);
        } finally {
            if(inputStream != null) {
                inputStream.close();
            }
        }

        pythonScript = "./" + new Path(globalConf.get(GlobalConfigurationKeys.PYTHON_SCRIPT_PATH)).getName();
        pythonShell = "./" + new Path(globalConf.get(GlobalConfigurationKeys.PYTHON_SHELL_PATH)).getName();
        HdfsUtils.givePerms(HDFSUtils.getLocalFS(), new File(pythonShell), true);

        // Since there is backup workers in cluster, we need this to get real worker number
        int numInstances = globalConf.getInt(GlobalConfigurationKeys.getInstancesKey(Constants.WORKER_JOB_NAME),
                GlobalConfigurationKeys.getDefaultInstances(Constants.WORKER_JOB_NAME));
        shellEnv.put("WORKER_CNT", Integer.toString(numInstances));
    }

    private int executeBackupExecutor() throws IOException, InterruptedException {
        String pythonScriptDst = "." + Constants.BACKUP_SCRIPT;
        shellEnv.put("TRAIN_SCRIPT_PATH", pythonScriptDst);
        shellEnv.put("IS_BACKUP", "True");

        tensorflowSocket.close();
        backupProcess = CommonUtils.executeShellAndGetProcess(pythonShell, shellEnv);
        backupProcess.waitFor();

        // 137 mean killed by ourselves
        if(backupProcess.exitValue() != 0 && backupProcess.exitValue() != 137) {
            LOG.info("backup task waiting process fails");
        }

        return backupProcess.exitValue();
    }

    public int run() throws IOException, InterruptedException {
        shellEnv.put("TRAIN_SCRIPT_PATH", pythonScript);

        // This is for this is backup task and replace slow task
        // It may take some time for appmaster to send training data to this backup task
        if(Constants.WORKER_JOB_NAME.equalsIgnoreCase(jobName)) {
            while(!shellEnv.containsKey("TRAINING_DATA_PATH")
                    || StringUtils.isBlank(shellEnv.get("TRAINING_DATA_PATH"))) {
                Thread.sleep(1000 * 5);
            }
        }

        if(!tensorflowSocket.isClosed()) {
            tensorflowSocket.close();
        }

        return CommonUtils.executeShell(pythonShell, 0, shellEnv);
    }

    public static void main(String[] args) throws Exception {
        LOG.info("TaskExecutor is running..");

        TensorflowTaskExecutor executor = new TensorflowTaskExecutor();

        // extract from args
        executor.init(initOpts(args));

        // copy script from jar,
        executor.prepare();

        // Register to tensorflow cluster via zookeeper
        // After register, we will wait the other worker to do so and get final cluster settings from zookeeper
        executor.registeryToCluster();

        if(executor.isBackup()) {
            LOG.info("This is backup host..");
            // this python script only join tensorflow cluster
            // and standby for calling
            int exitValue = executor.executeBackupExecutor();
            LOG.info("backup task exit value: " + exitValue);

            // We need wait until kill all previous process and then start
            executor.backupStartingLatch.await();
        }

        int result = executor.run();

        LOG.info("current worker finish..");
        System.exit(result);
    }

    public boolean isBackup() {
        return isBackup;
    }

}
