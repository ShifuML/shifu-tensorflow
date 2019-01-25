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
package ml.shifu.shifu.core.yarn.appmaster;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.client.api.async.impl.NMClientAsyncImpl;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.AbstractLivelinessMonitor;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;

import ml.shifu.guagua.coordinator.zk.GuaguaZooKeeper;
import ml.shifu.guagua.coordinator.zk.ZooKeeperUtils;
import ml.shifu.shifu.core.hadoop.MonotonicClock;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.Constants;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;
import ml.shifu.shifu.core.yarn.util.HdfsUtils;
import ml.shifu.shifu.util.HDFSUtils;

/**
 * @author webai
 *
 */
public class TensorflowApplicationMaster {
    private static final Log LOG = LogFactory.getLog(TensorflowApplicationMaster.class);

    /** HeartBeat monitor **/
    private final AbstractLivelinessMonitor<TensorflowTask> hbMonitor;
    private int hbInterval;
    private int maxConsecutiveHBMiss;
    private volatile boolean taskHasMissesHB = false;
    private Thread mainThread;

    /** Configuration **/
    private YarnConfiguration yarnConf = new YarnConfiguration();
    private Configuration globalConf = new Configuration();
    FileSystem hdfs = HDFSUtils.getFS();

    /** Tensorflow session **/
    private TensorflowSession session; // Create a dummy session for single node training.

    /** The environment set up for the TaskExecutor **/
    private Map<String, String> containerEnv = new ConcurrentHashMap<String, String>();

    /** Node manager delegates **/
    private NMCallbackHandler containerListener;

    AMRMCallbackHandler allocListener;
    
    private int appTimeout;
    private long workerTimeout;
    private int amRetryCount;
    private ContainerId containerId;
    private String appIdString;

    private String trainingDataPath;

    private TensorflowApplicationMaster() {
        hbMonitor = new AbstractLivelinessMonitor<TensorflowTask>("Tensorflow Task liveliness Monitor",
                new MonotonicClock()) {
            @Override
            protected void expire(TensorflowTask task) {
                onTaskDeemedDead(task);
            }

            @Override
            protected void serviceStart() throws Exception {
                setMonitorInterval(hbInterval * 3);
                setExpireInterval(hbInterval * Math.max(3, maxConsecutiveHBMiss)); // Be at least == monitoring interval
                super.serviceStart();
            }
        };
    }

    /**
     * Parse command line options and initialize TensorflowApplicationMaster
     * 
     * @return whether the initialization is successful or not.
     */
    private boolean init(String[] args) {
        globalConf.addResource(new Path(Constants.GLOBAL_FINAL_XML));

        hbMonitor.init(globalConf);

        try {
            Options opts = new Options();
            opts.addOption("container_env", true, "");
            CommandLine cliParser = new GnuParser().parse(opts, args);
            String[] containerEnvs = cliParser.getOptionValues("container_env");
            containerEnv.putAll(CommonUtils.parseKeyValue(containerEnvs));

            trainingDataPath = globalConf.get(GlobalConfigurationKeys.TRAINING_DATA_PATH);
        } catch (ParseException e) {
            LOG.error("Got exception while parsing options", e);
            return false;
        }

        Map<String, String> envs = System.getenv();
        appTimeout = globalConf.getInt(GlobalConfigurationKeys.APPLICATION_TIMEOUT,
                GlobalConfigurationKeys.DEFAULT_APPLICATION_TIMEOUT);
        workerTimeout = globalConf.getInt(GlobalConfigurationKeys.WORKER_TIMEOUT,
                GlobalConfigurationKeys.DEFAULT_WORKER_TIMEOUT);
        amRetryCount = globalConf.getInt(GlobalConfigurationKeys.AM_RETRY_COUNT,
                GlobalConfigurationKeys.DEFAULT_AM_RETRY_COUNT);
        containerId = ConverterUtils.toContainerId(envs.get(ApplicationConstants.Environment.CONTAINER_ID.name()));
        appIdString = containerId.getApplicationAttemptId().getApplicationId().toString();
        hbInterval = globalConf.getInt(GlobalConfigurationKeys.TASK_HEARTBEAT_INTERVAL_MS,
                GlobalConfigurationKeys.DEFAULT_TASK_HEARTBEAT_INTERVAL_MS);
        maxConsecutiveHBMiss = globalConf.getInt(GlobalConfigurationKeys.TASK_MAX_MISSED_HEARTBEATS,
                GlobalConfigurationKeys.DEFAULT_TASK_MAX_MISSED_HEARTBEATS);
        
        return true;
    }

    /**
     * Prepare the application master. This part is shared across different retries.
     */
    private boolean prepare() {
        LOG.info("Preparing application master..");

        this.session = new TensorflowSession(globalConf);
        
        containerListener = new NMCallbackHandler();
        nmClientAsync = new NMClientAsyncImpl(containerListener);
        nmClientAsync.init(yarnConf);
        nmClientAsync.start();

        // Init AMRMClient
        allocListener = new AMRMCallbackHandler(this.globalConf, this.session,
                this.nmClientAsync, this.hbMonitor, this.containerEnv, appIdString);
        amRMClient = AMRMClientAsync.createAMRMClientAsync(1000, allocListener);
        amRMClient.init(yarnConf);
        amRMClient.start();

        String hostname = CommonUtils.getCurrentHostName();

        try {
            amRMClient.registerApplicationMaster(hostname, -1, null);
        } catch (Exception e) {
            LOG.error("Exception while preparing AM", e);
            return false;
        }

        hbMonitor.start();

        return true;
    }


    private boolean run(String[] args) {
        if(!init(args)) {
            return false;
        }

        if(!prepare()) {
            return false;
        }

        mainThread = Thread.currentThread();

        boolean succeeded;
        do {
            try {
                // start the training job by building tensorflow session and schedule tasks by amRMClient
                this.session.scheduleTasks(amRMClient);
            } catch (Exception e) {
                LOG.error("Exception when we're starting TonyAM", e);
                return false;
            }

            succeeded = monitor();
            if(succeeded || amRetryCount == 0) {
                LOG.info("Result: " + succeeded + ", retry count: " + amRetryCount);
                break;
            }

            // Prepare for retryCount.
            try {
                reset();
            } catch (Exception e) {
                LOG.error("Error when reset.", e);
            }
            LOG.info("Retrying, remaining retry count" + amRetryCount);
            amRetryCount -= 1;
        } while(true);
        // Wait for the worker nodes to finish (The interval between registering the exit code to final exit)
        stop();

        return succeeded;
    }

    /**
     * Monitor the TensorFlow training job.
     * 
     * @return if the tensorflow job finishes successfully.
     */
    private boolean monitor() {
        long expireTime = appTimeout == 0 ? Long.MAX_VALUE : System.currentTimeMillis() + appTimeout;
        int counter = 0;
        while(true) {
            counter += 1;
            // Checking timeout
            if(System.currentTimeMillis() > expireTime) {
                LOG.error("Application times out.");
                break;
            }

            if(!session.isChiefWorkerSuccess()) {
                LOG.info("Chief Worker exist with non-zero exit code. Training has finished.");
                break;
            }

            if(this.taskHasMissesHB) {
                LOG.info("Application failed due to missed heartbeats");
                break;
            }

            if(this.session.getNumCompletedWorkerTasks().get() == this.session.getNumTotalWorkerTasks()) {
                CommonUtils.printWorkerTasksCompleted(this.session.getNumCompletedWorkerTasks(),
                        this.session.getNumTotalWorkerTasks());
                break;
            }

            // Reduce logging frequency to every 100s.
            if(counter % 20 == 1) {
                CommonUtils.printWorkerTasksCompleted(this.session.getNumCompletedWorkerTasks(),
                        this.session.getNumTotalWorkerTasks());
            }

            // Pause before refresh job status
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                LOG.error("Monitor: Thread interrupted", e);
            }
        }

        session.updateSessionStatus();

        CommonUtils.printWorkerTasksCompleted(this.session.getNumCompletedWorkerTasks(),
                this.session.getNumTotalWorkerTasks());

        FinalApplicationStatus status = session.getFinalStatus();
        String appMessage = session.getFinalMessage();
        if(status != FinalApplicationStatus.SUCCEEDED) {
            LOG.info("tensorflow session failed: " + appMessage);
        }
        return status == FinalApplicationStatus.SUCCEEDED;
    }

    // Reset state to prepare for retryCount.
    private void reset() {
        session.stopAllTasks(this.nmClientAsync);

        TensorflowSession.sessionId += 1;
        
        this.session = new TensorflowSession(this.globalConf);
        allocListener.setTensorflowSession(this.session);
        taskHasMissesHB = false;
    }

    private void stop() {
        FinalApplicationStatus status = session.getFinalStatus();
        String appMessage = session.getFinalMessage();
        try {
            amRMClient.unregisterApplicationMaster(status, appMessage, null);
        } catch (Exception ex) {
            LOG.error("Failed to unregister application", ex);
        }
        nmClientAsync.stop();
        amRMClient.waitForServiceToStop(5000);
        amRMClient.stop();

        // Pause before refresh job status
        try {
            Thread.sleep(30000);
        } catch (InterruptedException e) {
            LOG.error("stop: Thread interrupted", e);
        }
        
    }

    /**
     * Entry point of TensorflowApplicationMaster
     * The workflow of a training job in AM
     * prepare -> start -> failed -> reset -> retry if amRetryCount > 0 otherwise fail the job.
     * -> succeeded -> stop -> job succeeded
     * 
     * @param args
     *            the args from user inputs
     */
    public static void main(String[] args) {
        TensorflowApplicationMaster am = new TensorflowApplicationMaster();
        boolean succeeded = am.run(args);
        if(succeeded) {
            LOG.info("Application Master completed successfully. Exiting");
            System.exit(0);
        } else {
            LOG.info("Application Master failed. Exiting");
            System.exit(-1);
        }
    }

    private void onTaskDeemedDead(TensorflowTask task) {
        LOG.info("Task with id [" + task.getId() + "] has missed" + " [" + maxConsecutiveHBMiss
                + "] heartbeats.. Ending application !!");
        // TODO: figure out what is the right thing to do here..
        // TODO: For the time being, we just kill the job..
        String msg = "Task with id [" + task.getId() + "] deemed dead!!";
        LOG.error(msg);
        taskHasMissesHB = true;
        session.setFinalStatus(FinalApplicationStatus.FAILED, msg);
        mainThread.interrupt();
    }
}
