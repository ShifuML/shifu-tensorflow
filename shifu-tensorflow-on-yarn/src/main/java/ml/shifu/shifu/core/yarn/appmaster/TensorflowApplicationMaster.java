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

import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.impl.NMClientAsyncImpl;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.AbstractLivelinessMonitor;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.ZooDefs.Ids;

import ml.shifu.shifu.core.yarn.appmaster.TensorflowSession.SessionState;
import ml.shifu.shifu.core.yarn.appmaster.TensorflowSession.TensorflowClusterSpec;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.Constants;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;
import ml.shifu.shifu.util.HDFSUtils;

/**
 * {@link TensorflowApplicationMaster} is application master to launch ps and worker tasks.
 * 
 * <p>
 * This app master is used to check and launch all tasks, not to run the master task of distributed training. Master
 * task is run on another container.
 */
public class TensorflowApplicationMaster extends AbstractApplicationMaster {
    private static final Log LOG = LogFactory.getLog(TensorflowApplicationMaster.class);

    /** HeartBeat monitor **/
    private final AbstractLivelinessMonitor<TensorflowTask> hbMonitor;
    private int hbInterval;
    private int maxConsecutiveHBMiss;
    private volatile boolean taskHasMissesHB = false;

    /** Configuration **/
    private YarnConfiguration yarnConf = new YarnConfiguration();
    private Configuration globalConf = new Configuration();
    FileSystem hdfs = HDFSUtils.getFS();

    /** Tensorflow session **/
    private TensorflowSession session; // Create a dummy session for single node training.

    /** The environment set up for the TaskExecutor **/
    private Map<String, String> containerEnv = new ConcurrentHashMap<String, String>();
    
    private int appTimeout;
    private ContainerId containerId;
    private String appIdString;

    private TensorflowApplicationMaster() {
        // In order to improve performace of PS. https://blog.csdn.net/omnispace/article/details/79864973
        yarnConf.set("yarn.nodemanager.admin-env", "");
        
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
            
            private void onTaskDeemedDead(TensorflowTask task) {
                LOG.info("Task with id [" + task.getId() + "] has missed" + " [" + maxConsecutiveHBMiss
                        + "] heartbeats.. Ending application !!");
                // TODO: figure out what is the right thing to do here..
                // TODO: For the time being, we just kill the job..
                String msg = "Task with id [" + task.getId() + "] deemed dead!!";
                LOG.error(msg);
                taskHasMissesHB = true;
                //session.setFinalStatus(FinalApplicationStatus.FAILED, msg);
                //mainThread.interrupt();
            }
        };
    }

    /**
     * Entry point of TensorflowApplicationMaster
     * The workflow of a training job in AM
     * 
     * @param args
     *            the args from user inputs
     */
    public static void main(String[] args) {
        TensorflowApplicationMaster am = new TensorflowApplicationMaster();
        try {
            am.run(args);
            
            LOG.info("Application Master completed successfully. Exiting");
            System.exit(0);
        } catch (Exception e) {
            LOG.error("Fail to execute Tensorflow application master", e);
            System.exit(-1);
        }
    }

    /* 
     * Parse command line options and initialize TensorflowApplicationMaster
     */
    @Override
    protected void init(String[] args) {
        // retrieve information from args
        try {
            Options opts = new Options();
            opts.addOption("container_env", true, "");
            CommandLine cliParser = new GnuParser().parse(opts, args);
            containerEnv.putAll(CommonUtils.parseKeyValue(cliParser.getOptionValues("container_env")));
        } catch (ParseException e) {
            throw new IllegalStateException("Parsing app master arguments fails", e); 
        }

        // retrieve information from environment
        Map<String, String> envs = System.getenv();
        containerId = ConverterUtils.toContainerId(envs.get(ApplicationConstants.Environment.CONTAINER_ID.name()));
        appIdString = containerId.getApplicationAttemptId().getApplicationId().toString();
        
        // retrieve information from global config
        globalConf.addResource(new Path(Constants.GLOBAL_FINAL_XML));
        appTimeout = globalConf.getInt(GlobalConfigurationKeys.APPLICATION_TIMEOUT,
                GlobalConfigurationKeys.DEFAULT_APPLICATION_TIMEOUT);
        hbInterval = globalConf.getInt(GlobalConfigurationKeys.TASK_HEARTBEAT_INTERVAL_MS,
                GlobalConfigurationKeys.DEFAULT_TASK_HEARTBEAT_INTERVAL_MS);
        maxConsecutiveHBMiss = globalConf.getInt(GlobalConfigurationKeys.TASK_MAX_MISSED_HEARTBEATS,
                GlobalConfigurationKeys.DEFAULT_TASK_MAX_MISSED_HEARTBEATS);
        
        hbMonitor.init(globalConf);
        session = new TensorflowSession(globalConf, yarnConf);
    }

    @Override
    protected void registerRMCallbackHandler() {
        // Init AMRMClient
        AMRMCallbackHandler allocListener = new AMRMCallbackHandler(this.globalConf, this.session,
                this.nmClientAsync, this.hbMonitor, this.containerEnv, appIdString);
        amRMClient = AMRMClientAsync.createAMRMClientAsync(1000, allocListener);
        amRMClient.init(yarnConf);
        amRMClient.start();
    }

    /* 
     * Register RM callback and start listening
     */
    @Override
    protected void registerNMCallbackHandler() {
        NMCallbackHandler containerListener = new NMCallbackHandler();
        nmClientAsync = new NMClientAsyncImpl(containerListener);
        nmClientAsync.init(yarnConf);
        nmClientAsync.start();
    }

    @Override
    protected void prepareBeforeTaskExector() {
        hbMonitor.start();
    }
    
    /*
     * @see ml.shifu.shifu.core.yarn.appmaster.AbstractApplicationMaster#scheduleTask()
     */
    @Override
    protected void scheduleTask() {
        session.scheduleTasks(amRMClient); 
    }

    /* 
     * Monitor the TensorFlow training job.
     * 
     * @return if the tensorflow job finishes successfully.
     */
    @Override
    protected boolean monitor() {
        long expireTime = appTimeout == 0 ? Long.MAX_VALUE : System.currentTimeMillis() + appTimeout;
        while(true) {
            if (session.getState() == SessionState.STARTING_CONTAINER || 
                    session.getState() == SessionState.REGESTERING_CLUSTER) {
                // In order to prevent program waiting long time for one or two slow container starting,
                // here we do some logic to ingore those small number of slow container and starting training with whatever we 
                // have now.
                LOG.info("Session is REGESTERING_CLUSTER, we do check.");
                int readyPsCnt = session.getTensorflowClusterSpec().getReadyPsCnt();
                int readyWorkerCnt = session.getTensorflowClusterSpec().getReadyWorkerCnt();
                int totalPsCnt = session.getNumTotalPsTasks();
                int totalWorkerCnt = session.getNumTotalBackupWorkerTask() + session.getNumTotalWorkerTasks();
                boolean isChiefWorkerReady = session.getTensorflowClusterSpec()._isChiefWorkerReady();
                
                LOG.warn("readyPsCnt:" + readyPsCnt +
                        "totalPsCnt: " + totalPsCnt +
                        "readyWorkerCnt: " + readyWorkerCnt + 
                        "totalWorkerCnt: " + totalWorkerCnt);
                
                // if all ps and 95% of workers are ready and waiting time over 10 minuetes, we will continue training and 
                //  abandon those are not ready
                if (session.getState() == SessionState.REGESTERING_CLUSTER &&
                        isChiefWorkerReady &&
                        readyPsCnt > totalPsCnt * Constants.MIN_PS_START_TRAINING_THREASHOLD &&
                        readyWorkerCnt > totalWorkerCnt * Constants.MIN_WORKERS_START_TRAINING_THREASHOLD &&
                        (totalWorkerCnt - readyWorkerCnt) < session.getNumTotalBackupWorkerTask() && 
                        (System.currentTimeMillis() - session.getStartTimeOfRegisteringCluster()) > 
                            Constants.TIMEOUT_WAITING_CLUSTER_REGISTER) {
                    LOG.warn("We wait cluster register too long time, "
                            + "we are going to ignore worker cnt: " + (totalWorkerCnt-readyWorkerCnt) +
                            " and ignore ps cnt: " + (totalPsCnt - readyPsCnt));
                    
                    // we use every worker cluster now, and start training. do not accept any other workers
                    session.setState(SessionState.TRAINING);
                    TensorflowClusterSpec cluster = session.getTensorflowClusterSpec();
                    
                    // re-arrange PS to use back ps to replace ignoring one
                    int psBackCursor = totalPsCnt - 1;
                    for (int i = 0; i < psBackCursor; i++) {
                        if (StringUtils.isBlank(cluster.getPs()[i])) {
                            while(StringUtils.isBlank(cluster.getPs()[psBackCursor])) {
                                psBackCursor -= 1;
                            }
                            
                            if (psBackCursor > i) {
                                LOG.info("we are going to use ps task: " + psBackCursor + " to replace " + i);
                                
                                TensorflowTask replacement = 
                                        session.getTaskFromNormalTasks(Constants.PS_JOB_NAME, Integer.toString(psBackCursor));
                                cluster.getPs()[i] = cluster.getPs()[psBackCursor];
                                cluster.getPs()[psBackCursor] = null;
                                
                                TensorflowTask target = session.getTaskFromNormalTasks(Constants.PS_JOB_NAME, Integer.toString(i));
                                session.getJobNameToTasks().get(Constants.PS_JOB_NAME)[i] = replacement;

                                // remove replacement from backup list because it has new place already
                                replacement.setArrayIndex(target.getArrayIndex());
                                replacement.setTaskIndex(target.getTaskIndex());
                                
                                session.stopContainer(nmClientAsync, target.getContainer());
                                
                                psBackCursor -= 1;
                            }
                        }
                    }
                    
                    // re-arrange tensorflowCluster and task list to make sure order of host is same with task id
                    int workerBackCursor = totalWorkerCnt - 1;
                    for (int i = 0; i < workerBackCursor; i++) {
                        // TODO: if some worker fails, we need to use backup to replace it as well.
                        // Because during starting, some container are not started yet, some containers have fails already.
                        // so we need to deal with those registered failed workers
                        
                        if (StringUtils.isBlank(cluster.getWorker()[i])) {
                            while(StringUtils.isBlank(cluster.getWorker()[workerBackCursor])) {
                                workerBackCursor -= 1;
                            }
                            if (workerBackCursor > i) {
                                LOG.info("we are going to use worker task: " + workerBackCursor + " to replace " + i);
                                
                                // use back end one to replace front missing one, backup definetly be backup worker
                                // Otherwise it means all backup worker are dead which does not happened
                                TensorflowTask replacement = 
                                        session.getTaskFromBackupTasks(Constants.WORKER_JOB_NAME, Integer.toString(workerBackCursor));
                                cluster.getWorker()[i] = cluster.getWorker()[workerBackCursor];
                                cluster.getWorker()[workerBackCursor] = null;
                                
                                TensorflowTask target = null;
                                if (i+1 > session.getNumTotalWorkerTasks()) {
                                    // this is backup worker
                                    target = session.getTaskFromBackupTasks(Constants.WORKER_JOB_NAME, Integer.toString(i));
                                    // remove this from backup list from session
                                    session.getJobNameToBackupTask().get(Constants.WORKER_JOB_NAME).remove(target);
                                } else {
                                    target = session.getTaskFromNormalTasks(Constants.WORKER_JOB_NAME, Integer.toString(i));
                                    // replace replacement in target place
                                    session.getJobNameToTasks().get(Constants.WORKER_JOB_NAME)[i] = replacement;
                                    
                                    // weakup backup worker
                                    session.weakupBackup(replacement, target.getTrainingDataPaths());
                                }
                                
                                // remove replacement from backup list because it has new place already
                                session.getJobNameToBackupTask().get(Constants.WORKER_JOB_NAME).remove(replacement);
                                replacement.setArrayIndex(target.getArrayIndex());
                                replacement.setTaskIndex(target.getTaskIndex());
                                replacement.setTrainingDataPaths(target.getTrainingDataPaths());
                                
                                session.stopContainer(nmClientAsync, target.getContainer());
                                
                                workerBackCursor -= 1;
                            }
                        }
                    }
                    
                    // re-arrange array of cluster to remove null element
                    session.getJobNameToBackupTaskNum().put(Constants.WORKER_JOB_NAME, readyWorkerCnt-session.getNumTotalWorkerTasks());
                    cluster.setWorker(Arrays.copyOfRange(cluster.getWorker(), 0, readyWorkerCnt));
                    cluster.setPs(Arrays.copyOfRange(cluster.getPs(), 0, readyPsCnt));
                    
                    LOG.info("Left Backup worker : " + (readyWorkerCnt-session.getNumTotalWorkerTasks()));
                    
                    try {
                        TensorflowSession.getZookeeperServer().createOrSetExt(Constants.TENSORFLOW_FINAL_CLUSTER,
                                session.getTensorflowClusterSpec().toString().getBytes(Charset.forName("UTF-8")), 
                                Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT, true, -1);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
                
                if ((System.currentTimeMillis() - session.getStartTimeOfRegisteringCluster()) > (20 * 60 * 1000)) {
                    LOG.error("Wait too long for registering cluster. Please restart training....");
                    return true;
                }
            }
            
            
            // Checking timeout
            if(System.currentTimeMillis() > expireTime) {
                LOG.error("Application times out.");
                break;
            }

            if(!session.isChiefWorkerSuccess()) {
                LOG.info("Chief Worker exist with non-zero exit code. Training has finished.");
                break;
            }

            if(taskHasMissesHB) {
                LOG.info("Application failed due to missed heartbeats");
                break;
            }

            if (session.getFailedWorkers().size() > 0) {
                LOG.info("Some workers fails");
                break;
            }
            
            if (session.getFailedPs().size() >= session.getNumTotalPsTasks()) {
                LOG.info("All PS fails, training could not continue..");
                break;
            }
            
            if(session.isChiefWorkerComplete()) {
                LOG.info("Chief worker complete and success, so training process is over...");
                return true;
            }
            
            // Pause before refresh job status
            try {
                Thread.sleep(5000);
            } catch (InterruptedException e) {
                LOG.error("Monitor: Thread interrupted", e);
            }
        }

        return false;
    }
    
    @Override
    protected boolean canRecovered() {
        // is chief worker failed, whole jobs cannot continue
        if (!session.isChiefWorkerSuccess()) {
            return false;
        }
        
        // if any ps fails, cannot recover
        if (session.getFailedPs().size() >= session.getNumTotalPsTasks()) {
            return false;
        }
        
        // if worker failed number bigger than left backup worker plus tolerance
        if (session.getFailedWorkers().size() >= session.failedWorkerMaxLimit()) {
            return false;
        }
            
        return true;
    }
    
    @Override
    protected void recovery() {
        // we do not need to recover ps because backup ps and ps are same
        ConcurrentLinkedQueue<Integer> workerFailedQueue = session.getFailedWorkers();
        ConcurrentLinkedQueue<TensorflowTask> backupWorkerQueue = session.getJobNameToBackupTask()
                .get(Constants.WORKER_JOB_NAME);
        while(!workerFailedQueue.isEmpty() && !backupWorkerQueue.isEmpty()) {
            TensorflowTask backupWorkerTask = backupWorkerQueue.poll();
            Integer failedWorkerTaskArrayId = workerFailedQueue.poll();
            
            try {
                session.weakupBackup(backupWorkerTask, failedWorkerTaskArrayId);
            } catch (Exception e) {
                LOG.error("error to write zookeeper", e);
            }
        }
    }    
    @Override
    protected void updateTaskStatus() {
        session.updateSessionStatus();

        CommonUtils.printWorkerTasksCompleted(this.session.getNumCompletedWorkerTasks(),
                this.session.getNumTotalWorkerTasks());

        FinalApplicationStatus status = session.getFinalStatus();
        String appMessage = session.getFinalMessage();
        if(status != FinalApplicationStatus.SUCCEEDED) {
            LOG.info("tensorflow session failed: " + appMessage);
        } else {
            LOG.info("tensorflow session is successful");
        }
    }

    @Override
    protected void stop() {
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
}
