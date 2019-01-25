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
import java.net.ServerSocket;
import java.net.UnknownHostException;
import java.nio.charset.Charset;
import java.nio.file.CopyOption;
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
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooDefs.Ids;

import ml.shifu.guagua.coordinator.zk.GuaguaZooKeeper;
import ml.shifu.guagua.coordinator.zk.ZooKeeperUtils;
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

    /** Use for wait all workers registering on Master **/
    final CountDownLatch latch = new CountDownLatch(1);

    private String tensorflowCluster;
    /** Use for reserve port for tensorflow cluster **/
    private ServerSocket tensorflowSocket;

    private Configuration globalConf = new Configuration();

    private Map<String, String> shellEnv = new HashMap<String, String>();

    public TensorflowTaskExecutor()  {
        globalConf.addResource(new Path(Constants.GLOBAL_FINAL_XML));
    }

    public void registeryToCluster() throws KeeperException, InterruptedException, IOException {
        String port = getTensorflowPort();

        if(StringUtils.isBlank(port)) {
            throw new RuntimeException("Given port on container is blank!");
        }

        // Keep watching final cluster. if master write final cluster, it means workers could continue training script
        zookeeper.exists(Constants.TENSORFLOW_FINAL_CLUSTER, true);

        zookeeper.createOrSetExt(Constants.TENSORFLOW_CLUSTER_ROOT_PATH + containerId,
                (CommonUtils.getCurrentHostIP() + ":" + port).getBytes(Charset.forName("UTF-8")), Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT, true, -1);

        latch.await();
    }

    // callback method would be trigger if master collect all workers ip and port
    public void process(WatchedEvent event) {
        LOG.info("event path:" + event.getPath());
        if(Watcher.Event.EventType.NodeCreated == event.getType() && 
                Constants.TENSORFLOW_FINAL_CLUSTER.equalsIgnoreCase(event.getPath())) {
            try {
                tensorflowCluster = new String(zookeeper.getData(Constants.TENSORFLOW_FINAL_CLUSTER, false, null));
                LOG.info("Cluster:" + tensorflowCluster);
                latch.countDown();
                zookeeper.close();
            } catch (Exception e) {
                LOG.error("Error when getting final cluster from zookeeper", e);
                throw new RuntimeException(e);
            }
        }
    }

    public String getTensorflowPort() throws IOException {
        tensorflowSocket = new ServerSocket(ZooKeeperUtils.getValidZooKeeperPort());
    
        return Integer.toString(this.tensorflowSocket.getLocalPort());
    }

    /**
     * 
     * @param env
     * @throws InterruptedException
     * @throws IOException
     */
    public void run() throws IOException, InterruptedException {
        this.shellEnv.put("CLUSTER_SPEC", this.tensorflowCluster);// json{'ps':[1.1.1.1:123,1.1.1.2:123],'worker':[1.1.1.1:123]}
        this.shellEnv.put("WEIGHT_COLUMN_NUM", globalConf.get(GlobalConfigurationKeys.WEIGHT_COLUMN_NUM,
                GlobalConfigurationKeys.DEFAULT_WEIGHT_COLUMN_NUM)); // default is -1.
        this.shellEnv.put("MODEL_OUTPUT", "./models");
        
        // Copy shell from jar so that we could execute
        Files.copy(this.getClass().getResourceAsStream("/pytrain.sh"), 
                Paths.get("./pytrain.sh"), StandardCopyOption.REPLACE_EXISTING);

        HdfsUtils.givePerms(HDFSUtils.getLocalFS(), new File("./pytrain.sh"), true);

        tensorflowSocket.close();
        CommonUtils.executeShell("./pytrain.sh", 0, this.shellEnv);

    }

    public static CommandLine initOpts(String[] args) throws ParseException {
        Options opts = new Options();

        opts.addOption("job_name", true, "");
        opts.addOption("task_id", true, "");
        opts.addOption("container_id", true, "");
        opts.addOption("training_data_path", true, "");
        opts.addOption("zookeeper_server", true, "");
        
        return new GnuParser().parse(opts, args);
    }

    public void parseArgsIntoEnv(CommandLine cliParser) throws ParseException, IOException {
        // Use in train.py
        shellEnv.put("JOB_NAME", cliParser.getOptionValue("job_name")); // worker or ps
        shellEnv.put("TASK_ID", cliParser.getOptionValue("task_id"));
        shellEnv.put("TRAINING_DATA_PATH", cliParser.getOptionValue("training_data_path")); // /path/a,/path/b
        
        containerId = cliParser.getOptionValue("container_id").trim();

        String zookeeperServer = cliParser.getOptionValue("zookeeper_server");
        zookeeper = new GuaguaZooKeeper(zookeeperServer, 300000, 5, 1000, this);
    }

    public void parseConfIntoEnv() throws IOException {
        String pythonBinaryPath = globalConf.get(GlobalConfigurationKeys.PYTHON_BINARY_PATH);
        String glibcBinaryPath = globalConf.get(GlobalConfigurationKeys.GLIBC_BINARY_PATH);
        String pythonScriptPath = globalConf.get(GlobalConfigurationKeys.PYTHON_SCRIPT_PATH);
        String pythonScriptDst = "." + pythonScriptPath;
        Files.copy(this.getClass().getResourceAsStream(pythonScriptPath), 
                Paths.get(pythonScriptDst), StandardCopyOption.REPLACE_EXISTING);
        
        // Use in bash
        shellEnv.put("GLIBC_HOME", "." + glibcBinaryPath);
        shellEnv.put("PYTHON_HOME", "." + pythonBinaryPath);
        shellEnv.put("TRAIN_SCRIPT_PATH", pythonScriptDst);
    }

    public void prepare() {
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
    }

    public static void main(String[] args) throws Exception {
        LOG.info("TaskExecutor is running..");

        TensorflowTaskExecutor executor = new TensorflowTaskExecutor();

        // extract from args
        executor.parseArgsIntoEnv(initOpts(args));
        executor.parseConfIntoEnv();

        executor.prepare();
        // Register to tensorflow cluster via zookeeper
        // After register, we will wait the other worker to do so and get final cluster settings from zookeeper
        executor.registeryToCluster();

        executor.run();
    }
}
