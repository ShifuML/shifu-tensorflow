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

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.client.api.AMRMClient;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooDefs.Ids;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.google.common.base.Preconditions;

import ml.shifu.guagua.coordinator.zk.GuaguaZooKeeper;
import ml.shifu.guagua.coordinator.zk.ZooKeeperUtils;
import ml.shifu.shifu.core.TrainingIntermediateResult;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.Constants;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;

/**
 * Tensorflow Session contains all tensorflow jobs information
 * 
 * @author webai
 * 
 */
public class TensorflowSession implements Watcher {
    private static final Log LOG = LogFactory.getLog(TensorflowSession.class);
    private Configuration globalConf;

    private Map<String, TensorFlowContainerRequest> containerRequests;
    private Map<String, List<ContainerRequest>> jobNameToContainerRequests = new HashMap<String, List<ContainerRequest>>();
    private Map<String, TensorflowTask> containerIdToTask = new HashMap<String, TensorflowTask>();
    // A map from task name to an array of TFTasks with that name.
    private Map<String, TensorflowTask[]> jobNameToTasks = new ConcurrentHashMap<String, TensorflowTask[]>();
    private Map<String, ConcurrentLinkedQueue<TensorflowTask>> jobNameToBackupTask = new ConcurrentHashMap<String, ConcurrentLinkedQueue<TensorflowTask>>();

    private TensorflowClusterSpec tensorflowClusterSpec;

    /** those task not have container **/
    private Map<String, Integer> jobNameToPendingTaskNumber = new ConcurrentHashMap<String, Integer>();
    private Map<String, Integer> jobNameToPendingBackupTaskNumber = new ConcurrentHashMap<String, Integer>();
    private int numRequestedContainers = 0;

    /** train data set **/
    private List<StringBuilder> splitedTrainingData = null;

    /** Job progress **/
    private boolean isChiefWorkerComplete = false;
    private AtomicInteger numCompletedWorkerTasks = new AtomicInteger(0);
    private Map<String, Integer> jobNameToTaskNum = new ConcurrentHashMap<String, Integer>();
    private Map<String, Integer> jobNameToBackupTaskNum = new HashMap<String, Integer>();

    /** failed task index in task array including timeout task and return wrong exit code **/
    private ConcurrentLinkedQueue<Integer> failedWorkers = new ConcurrentLinkedQueue<Integer>();
    /** failed only when return wrong exit code, store task index of ps **/
    private ConcurrentLinkedQueue<String> failedPs = new ConcurrentLinkedQueue<String>();

    // sessionId to distinguish different sessions. Currently used to distinguish
    // failed session and new session.
    public static int sessionId = 0;

    private FinalApplicationStatus sessionFinalStatus = FinalApplicationStatus.UNDEFINED;
    private String sessionFinalMessage = null;

    // if Chief worker finished with non-zero exit code, we stop whole training
    private boolean chiefWorkerSuccess = true;

    private static String zookeeperServerHostPort = null;

    private static GuaguaZooKeeper zookeeperServer = null;

    private int totalEpochs;

    private AtomicInteger globalEpoch = new AtomicInteger(0);

    private ConcurrentHashMap<String, TrainingIntermediateResult> intermediateResults = new ConcurrentHashMap<String, TrainingIntermediateResult>();

    // Record session state for monitoring
    private SessionState state;

    private long startTimeOfRegisteringCluster;

    public TensorflowSession() {
        setState(SessionState.INIT);
    }

    public TensorflowSession(Configuration globalConf) {
        this.globalConf = globalConf;
        this.totalEpochs = this.globalConf.getInt(GlobalConfigurationKeys.SHIFU_APPLICATION_EPOCHS, -1);
        this.containerRequests = CommonUtils.parseContainerRequests(this.globalConf);
        setState(SessionState.INIT);

        // create zookeeper server for sync tensorflow cluster spec
        // This has been settled in prepare of AM
        if(zookeeperServer == null) {
            zookeeperServerHostPort = startZookeeperServer();
            try {
                zookeeperServer = new GuaguaZooKeeper(zookeeperServerHostPort, 3000000, 5, 1000, this);
            } catch (IOException e) {
                LOG.error("create zookeeper server fails!", e);
                throw new RuntimeException(e);
            }
        }

        for(Entry<String, TensorFlowContainerRequest> jobNameToContainerReq: containerRequests.entrySet()) {
            String jobName = jobNameToContainerReq.getKey();
            TensorFlowContainerRequest req = jobNameToContainerReq.getValue();

            int taskCnt = req.getNumInstances();
            int backupTaskCnt = req.getNumBackupInstances();

            jobNameToTasks.put(jobName, new TensorflowTask[taskCnt]);
            jobNameToBackupTask.put(jobName, new ConcurrentLinkedQueue<TensorflowTask>());

            jobNameToTaskNum.put(jobName, taskCnt);
            jobNameToBackupTaskNum.put(jobName, backupTaskCnt);
        }

        this.tensorflowClusterSpec = new TensorflowClusterSpec(
                jobNameToTaskNum.get(Constants.PS_JOB_NAME) + jobNameToBackupTaskNum.get(Constants.PS_JOB_NAME),
                jobNameToTaskNum.get(Constants.WORKER_JOB_NAME)
                        + jobNameToBackupTaskNum.get(Constants.WORKER_JOB_NAME));

        // Split training data for workers
        try {
            splitedTrainingData = TrainingDataSet.getInstance().getSplitedFilePaths(globalConf,
                    jobNameToTaskNum.get(Constants.WORKER_JOB_NAME),
                    globalConf.get(GlobalConfigurationKeys.TRAINING_DATA_PATH));
            LOG.info("splitedTrainingData: " + splitedTrainingData.toString());
        } catch (Exception e) {
            LOG.error("Splitting training data fails or count file line fails!", e);
            throw new RuntimeException(e);
        }
    }

    private static String startZookeeperServer() {
        String localHostName = CommonUtils.getCurrentHostName();

        int embedZkClientPort = 0;
        try {
            embedZkClientPort = ZooKeeperUtils.startEmbedZooKeeper();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // 2. check if it is started.
        ZooKeeperUtils.checkIfEmbedZooKeeperStarted(embedZkClientPort);
        return localHostName + ":" + embedZkClientPort;
    }

    public void scheduleTasks(AMRMClientAsync<ContainerRequest> amRMClient) {
        for(Entry<String, TensorFlowContainerRequest> jobNameToContainerReq: containerRequests.entrySet()) {
            String jobName = jobNameToContainerReq.getKey();
            TensorFlowContainerRequest containerRequest = jobNameToContainerReq.getValue();

            if(!jobNameToContainerRequests.containsKey(jobName)) {
                jobNameToContainerRequests.put(jobName, new ArrayList<ContainerRequest>());
            }

            // prepare resource request and add to amRMClient
            for(int i = 0; i < (containerRequest.getNumInstances() + containerRequest.getNumBackupInstances()); i++) {
                AMRMClient.ContainerRequest containerAsk = setupContainerRequestForRM(containerRequest);

                jobNameToContainerRequests.get(jobName).add(containerAsk);

                amRMClient.addContainerRequest(containerAsk);

                numRequestedContainers++;
            }

            jobNameToPendingTaskNumber.put(jobName, containerRequest.getNumInstances());
            jobNameToPendingBackupTaskNumber.put(jobName, containerRequest.getNumBackupInstances());
        }
        setState(SessionState.STARTING_CONTAINER);
    }

    private AMRMClient.ContainerRequest setupContainerRequestForRM(TensorFlowContainerRequest request) {
        Priority priority = Priority.newInstance(request.getPriority());
        Resource capability = Resource.newInstance((int) request.getMemory(), request.getVCores());
        AMRMClient.ContainerRequest containerRequest = new AMRMClient.ContainerRequest(capability, null, null,
                priority);
        LOG.info("Requested container ask: " + containerRequest.toString());
        return containerRequest;
    }

    /**
     * @param container
     * @return return null means error on pending task number
     */
    public synchronized TensorflowTask distributeTaskToContainer(Container container) {
        String jobName = getAvailableJobName(container);

        if(StringUtils.isBlank(jobName)) {
            LOG.error("Cannot distribute container: " + container.toString());
            throw new RuntimeException("couldn't find job to match container");
        } else {
            String containerId = container.getId().toString();
            try {
                // Use for collect worker ip and port for tensorflow cluster Construction
                zookeeperServer.exists(Constants.TENSORFLOW_CLUSTER_ROOT_PATH + containerId, this);
                // Use for collect worker intermiate result for log printing and monitor
                zookeeperServer.exists(Constants.WORKER_INTERMEDIATE_RESULT_ROOT_PATH + containerId, this);
            } catch (Exception e) {
                LOG.error("watch container fails", e);
                throw new RuntimeException(e);
            }

            if(jobNameToPendingTaskNumber.get(jobName) > 0) {
                // distribute container to task
                TensorflowTask[] tasks = jobNameToTasks.get(jobName);
                for(int i = 0; i < tasks.length; i++) {
                    if(tasks[i] == null) {
                        tasks[i] = new TensorflowTask(jobName, String.valueOf(i), sessionId, container,
                                zookeeperServerHostPort, globalConf, false, i);

                        if(Constants.WORKER_JOB_NAME.equalsIgnoreCase(jobName)) {
                            // Only worker has training data
                            tasks[i].setTrainingDataPaths(splitedTrainingData.get(i).toString());
                        }

                        jobNameToPendingTaskNumber.put(jobName, jobNameToPendingTaskNumber.get(jobName) - 1);
                        containerIdToTask.put(containerId, tasks[i]);
                        return tasks[i];
                    }
                }
            } else if(jobNameToPendingBackupTaskNumber.get(jobName) > 0) {
                // distribute container to backup task
                int taskId = jobNameToTaskNum.get(jobName) + jobNameToBackupTask.get(jobName).size();
                TensorflowTask task = new TensorflowTask(jobName, String.valueOf(taskId), sessionId, container,
                        zookeeperServerHostPort, globalConf, true, -1);

                jobNameToBackupTask.get(jobName).offer(task);
                jobNameToPendingBackupTaskNumber.put(jobName, jobNameToPendingBackupTaskNumber.get(jobName) - 1);
                containerIdToTask.put(containerId, task);
                return task;
            }
        }

        throw new RuntimeException("Error when distribute container to task");
    }

    /**
     * Available job need two requirements: 1. container resource(mem, core) is same as job request resource
     * 2. job task is not full
     * 
     * @param container
     * @return
     */
    private synchronized String getAvailableJobName(Container container) {
        LOG.info("allocated resource: " + container.getResource().toString());
        LOG.info("remaining resource: " + jobNameToPendingTaskNumber);
        LOG.info("session id: " + sessionId);
        for(Map.Entry<String, TensorFlowContainerRequest> jobNameToRequest: containerRequests.entrySet()) {
            String jobName = jobNameToRequest.getKey();
            TensorFlowContainerRequest request = jobNameToRequest.getValue();
            int pendingNumber = jobNameToPendingTaskNumber.get(jobName);
            int pendingBackupNumber = jobNameToPendingBackupTaskNumber.get(jobName);

            if((int) request.getMemory() == container.getResource().getMemory()
                    && request.getVCores() == container.getResource().getVirtualCores()
                    && (pendingNumber > 0 || pendingBackupNumber > 0)) {
                return jobName;
            }
        }

        return null;
    }

    public TensorflowTask getTaskByContainerId(ContainerId containerId) {
        return this.containerIdToTask.get(containerId.toString());
    }

    public TensorflowTask getTaskFromNormalTasks(String jobName, String taskIndex) {
        for(Map.Entry<String, TensorflowTask[]> entry: this.jobNameToTasks.entrySet()) {
            TensorflowTask[] tasks = entry.getValue();
            for(TensorflowTask task: tasks) {
                if(task.getJobName().equals(jobName) && task.getTaskIndex().equals(taskIndex)) {
                    return task;
                }
            }
        }

        return null;
    }

    public TensorflowTask getTaskFromBackupTasks(String jobName, String taskIndex) {
        Iterator<TensorflowTask> backupItr = jobNameToBackupTask.get(jobName).iterator();
        while(backupItr.hasNext()) {
            TensorflowTask task = backupItr.next();
            if(task.getJobName().equals(jobName) && task.getTaskIndex().equals(taskIndex)) {
                return task;
            }
        }
        return null;
    }

    private boolean isChief(String jobName, String jobIndex) {
        String chiefName = Constants.WORKER_JOB_NAME;
        String chiefIndex = "0";
        return jobName.equals(chiefName) && jobIndex.equals(chiefIndex);
    }

    public void setFinalStatus(FinalApplicationStatus status, String message) {
        sessionFinalStatus = status;
        sessionFinalMessage = message;
    }

    private int getFailedTasksNum(TensorflowTask[] tasks) {
        int failedTaskNum = 0;
        for(TensorflowTask task: tasks) {
            if(task == null) {
                String msg = "Job is null, this should not happen.";
                LOG.error(msg);
                setFinalStatus(FinalApplicationStatus.FAILED, msg);
                return 0;
            }
            if(task.exitStatus != 0) {
                failedTaskNum += 1;
                String msg = "Job " + task.getJobName() + " at index: " + task.getTaskIndex()
                        + " haven't finished yet.";
                LOG.error(msg);
            }
        }
        return failedTaskNum;
    }

    /**
     * To get max number of ps could failed and training process does not have impact
     * 
     * @return
     */
    public double failedPsMaxLimit() {
        return Constants.PS_FAULT_TOLERNANCE_THREAHOLD * getNumTotalPsTasks()
                + getJobNameToBackupTask().get(Constants.PS_JOB_NAME).size();
    }

    /**
     * To get max number of worker could failed and training process does not have impact
     * 
     * @return
     */
    public double failedWorkerMaxLimit() {
        return Constants.WORKER_FAULT_TOLERENANCE_THRESHOLD * getNumTotalWorkerTasks()
                + getJobNameToBackupTask().get(Constants.WORKER_JOB_NAME).size();
    }

    /**
     * Update the status of a session and set exit code if a session is completed.
     */
    public void updateSessionStatus() {
        if(getFinalStatus() == FinalApplicationStatus.FAILED) {
            return;
        }

        if(chiefWorkerSuccess) {
            LOG.info("Session completed with chief worker success, setting final status SUCCEEDED.");
            setFinalStatus(FinalApplicationStatus.SUCCEEDED, null);
        } else {
            setFinalStatus(FinalApplicationStatus.FAILED, "Chief worker failed");
        }

        /**
         * int failedWorkerNum = getFailedTasksNum(jobNameToTasks.get(Constants.WORKER_JOB_NAME));
         * int failedPsNum = getFailedTasksNum(jobNameToTasks.get(Constants.PS_JOB_NAME));
         * 
         * if(failedPsNum > 0) {
         * setFinalStatus(FinalApplicationStatus.FAILED, "There are some PS fails, they are: " + failedPs.toString());
         * } else if(failedWorkerNum >= failedWorkerMaxLimit()) {
         * setFinalStatus(FinalApplicationStatus.FAILED,
         * "More than threshold of worker failed, failedCnt=" + failedWorkerNum);
         * } else if(!chiefWorkerSuccess) {
         * setFinalStatus(FinalApplicationStatus.FAILED, "Chief worker failed");
         * } else {
         * LOG.info("Session completed with no job failures, setting final status SUCCEEDED.");
         * setFinalStatus(FinalApplicationStatus.SUCCEEDED, null);
         * }
         **/
    }

    /**
     * Refresh task status on each TaskExecutor registers its exit code with AM.
     */
    public void onTaskCompleted(String jobName, String jobIndex, int exitCode) {
        LOG.info(String.format("Job %s:%s exited with %d", jobName, jobIndex, exitCode));
        TensorflowTask backupTask = getTaskFromBackupTasks(jobName, jobIndex);
        if(backupTask != null) {
            // if backup task fails, we just remove it from standing-by queue
            // do not need to worry
            jobNameToBackupTask.get(jobName).remove(backupTask);
            LOG.error("backup task fails!!");
            return;
        }

        TensorflowTask task = getTaskFromNormalTasks(jobName, jobIndex);
        Preconditions.checkNotNull(task);

        // mark current task as completed
        task.setExitStatus(exitCode);

        if(Constants.WORKER_JOB_NAME.equals(jobName)) {
            LOG.info(jobName + ":" + jobIndex + ":" + exitCode);

            if(exitCode == 0) {
                // success
                numCompletedWorkerTasks.incrementAndGet();
                if(isChief(jobName, jobIndex)) {
                    isChiefWorkerComplete = true;
                }
            } else {
                if(isChief(jobName, jobIndex)) {
                    // If the chief worker failed[chief or worker 0], short circuit and stop the training. Note that
                    // even though other worker failures will also fail the job but we don't short circuit the training
                    // because the training can still continue, while if chief worker is dead, a TensorFlow training
                    // would hang. Also note that, we only short circuit when the chief worker failed, not finished.
                    chiefWorkerSuccess = false;
                }
                failedWorkers.offer(task.getArrayIndex());
            }
        } else {
            if(exitCode != 0) {
                // ps fails
                failedPs.offer(task.getTaskIndex());
            }
        }

    }

    public void stopAllTasks(NMClientAsync nmClientAsync) {
        for(TensorflowTask task: this.containerIdToTask.values()) {
            if(task != null) {
                nmClientAsync.stopContainerAsync(task.getContainer().getId(), task.getContainer().getNodeId());
                LOG.info("Stop a task in container: containerId = " + task.getContainer().getId() + ", containerNode = "
                        + task.getContainer().getNodeId().getHost());
            }
        }
    }

    public void stopContainer(NMClientAsync nmClientAsync, Container container) {
        LOG.info("container: " + container);
        LOG.info("Stop a task in container: containerId = " + container.getId() + ", containerNode = "
                + container.getNodeId().getHost());
        nmClientAsync.stopContainerAsync(container.getId(), container.getNodeId());
    }

    public boolean isChiefWorkerComplete() {
        return isChiefWorkerComplete;
    }

    public AtomicInteger getNumCompletedWorkerTasks() {
        return numCompletedWorkerTasks;
    }

    public void setNumCompletedWorkerTasks(AtomicInteger numCompletedWorkerTasks) {
        this.numCompletedWorkerTasks = numCompletedWorkerTasks;
    }

    public int getNumTotalBackupWorkerTask() {
        return jobNameToBackupTaskNum.get(Constants.WORKER_JOB_NAME);
    }

    public int getNumTotalWorkerTasks() {
        return jobNameToTaskNum.get(Constants.WORKER_JOB_NAME);
    }

    public int getNumTotalPsTasks() {
        return jobNameToTaskNum.get(Constants.PS_JOB_NAME);
    }

    public ConcurrentLinkedQueue<Integer> getFailedWorkers() {
        return failedWorkers;
    }

    public ConcurrentLinkedQueue<String> getFailedPs() {
        return failedPs;
    }

    public Map<String, Integer> getJobNameToBackupTaskNum() {
        return jobNameToBackupTaskNum;
    }

    private void doStatistic() {
        int count = intermediateResults.size();
        double trainingErrorSum = 0.0d;
        double validErrorSum = 0.0d;
        double trainingTimeSum = 0.0d;
        double validTimeSum = 0.0d;

        for(Entry<String, TrainingIntermediateResult> entrySet: intermediateResults.entrySet()) {
            TrainingIntermediateResult tir = entrySet.getValue();
            trainingErrorSum += tir.getTrainingError();
            validErrorSum += tir.getValidError();
            trainingTimeSum += tir.getCurrentEpochTime();
            validTimeSum += tir.getCurrentEpochValidTime();
        }
        LOG.info("Epoch: " + getGlobalEpoch().get() + " training error: " + (trainingErrorSum / count)
                + " valid error: " + (validErrorSum / count) + " training avg time: " + (trainingTimeSum / count)
                + " valid avg time: " + (validTimeSum / count));
    }

    public void process(WatchedEvent event) {
        if(event == null || StringUtils.isBlank(event.getPath())) {
            return;
        }

        if(event.getPath().contains(Constants.TENSORFLOW_CLUSTER_ROOT_PATH)) {
            // Collect worker ip and port for tensorflow cluster
            String containerId = event.getPath().replace(Constants.TENSORFLOW_CLUSTER_ROOT_PATH, "");
            TensorflowTask task = this.containerIdToTask.get(containerId);

            if(this.getState() != SessionState.REGESTERING_CLUSTER) {
                LOG.warn("Tensorflow session is not REGESTERING_CLUSTER currently but try to register, we will ingore "
                        + task);
                return;
            }

            LOG.info("This is tensorflow cluster...");
            try {
                String ipAndPort = new String(zookeeperServer.getData(event.getPath(), null, null));
                task.setRegister(true);
                tensorflowClusterSpec.add(task.getJobName(), Integer.valueOf(task.getTaskIndex()), ipAndPort);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            int readyContainersNumber = tensorflowClusterSpec.totalWorkerAndPs();
            if(readyContainersNumber == numRequestedContainers) {
                LOG.info("Get all tensorflow cluster host and port from containers");
                try {
                    zookeeperServer.createOrSetExt(Constants.TENSORFLOW_FINAL_CLUSTER,
                            tensorflowClusterSpec.toString().getBytes(Charset.forName("UTF-8")), Ids.OPEN_ACL_UNSAFE,
                            CreateMode.PERSISTENT, true, -1);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                // we give every worker cluster now, and start training.
                this.setState(SessionState.TRAINING);
            } else if(readyContainersNumber < numRequestedContainers) {
                LOG.info("Total requested containers: " + numRequestedContainers + ", ready containers: "
                        + readyContainersNumber);
            } else {
                LOG.error("Total requested containers: " + numRequestedContainers + ", ready containers: "
                        + readyContainersNumber);
            }
        } else if(event.getPath().contains(Constants.WORKER_INTERMEDIATE_RESULT_ROOT_PATH)) {
            // LOG.info("This is worker intermediate result...");
            String containerId = event.getPath().replace(Constants.WORKER_INTERMEDIATE_RESULT_ROOT_PATH, "");
            synchronized(TensorflowSession.class) {
                try {
                    TrainingIntermediateResult intermediateResult = new TrainingIntermediateResult(
                            zookeeperServer.getData(event.getPath(), this, null));
                    int currentEpoch = intermediateResult.getCurrentEpochStep();
                    if(currentEpoch < getGlobalEpoch().get()) {
                        // ignore stale information
                    } else if(currentEpoch == getGlobalEpoch().get()) {
                        intermediateResults.putIfAbsent(containerId, intermediateResult);
                    } else {
                        // this is the start of next epoch, so we need to calculate current epoch
                        LOG.info("Epoch " + getGlobalEpoch() + " is finish..");
                        doStatistic();

                        // do clean
                        intermediateResults.clear();
                        getGlobalEpoch().set(currentEpoch);

                        // add current into new map for next epoch
                        intermediateResults.putIfAbsent(containerId, intermediateResult);
                    }

                } catch (Exception e) {
                    LOG.error("Getting worker intermediate result fails.", e);
                    throw new RuntimeException(e);
                }
            }

        }
    }

    /**
     * If there is no pending task, it means all tasks have container
     * 
     * @return
     */
    public boolean isAllTaskAssignedContainer() {
        int sum = 0;
        for(int i: jobNameToPendingTaskNumber.values()) {
            sum += i;
        }

        for(int i: jobNameToPendingBackupTaskNumber.values()) {
            sum += i;
        }

        return (sum == 0);
    }

    public boolean isChiefWorkerSuccess() {
        return chiefWorkerSuccess;
    }

    public FinalApplicationStatus getFinalStatus() {
        return this.sessionFinalStatus;
    }

    public String getFinalMessage() {
        return this.sessionFinalMessage;
    }

    public Configuration getGlobalConf() {
        return this.globalConf;
    }

    public Map<String, TensorflowTask[]> getJobNameToTasks() {
        return jobNameToTasks;
    }

    public Map<String, ConcurrentLinkedQueue<TensorflowTask>> getJobNameToBackupTask() {
        return jobNameToBackupTask;
    }

    class TensorflowClusterSpec {
        // In order to make spec host order same as task order
        private String[] ps;
        private String[] worker;
        private int readyPsCnt = 0;
        private int readyWorkerCnt = 0;
        private boolean isChiefWorkerReady = false;

        TensorflowClusterSpec(int psTaskCnt, int workerTaskCnt) {
            ps = new String[psTaskCnt];
            worker = new String[workerTaskCnt];
        }

        public synchronized void add(String jobName, int taskId, String hostnamePort) {
            if("ps".equalsIgnoreCase(jobName)) {
                ps[taskId] = hostnamePort;
                readyPsCnt += 1;
            } else {
                worker[taskId] = hostnamePort;
                readyWorkerCnt += 1;
            }

            if(Constants.WORKER_JOB_NAME.equals(jobName) && taskId == 0) {
                isChiefWorkerReady = true;
            }
        }

        public String[] getPs() {
            return ps;
        }

        public String[] getWorker() {
            return worker;
        }

        public void setWorker(String[] worker) {
            this.worker = worker;
        }

        public int _getReadyPsCnt() {
            return readyPsCnt;
        }

        public int _getReadyWorkerCnt() {
            return readyWorkerCnt;
        }

        public synchronized int totalWorkerAndPs() {
            return readyWorkerCnt + readyPsCnt;
        }

        public String toString() {
            ObjectWriter ow = new ObjectMapper().writer().withDefaultPrettyPrinter();
            try {
                return ow.writeValueAsString(this);
            } catch (JsonProcessingException e) {
                LOG.error("transfer cluster failed", e);
            }
            // return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
            return StringUtils.EMPTY;
        }

        public boolean _isChiefWorkerReady() {
            return isChiefWorkerReady;
        }
    }

    /**
     * @param backupWorkerTask
     * @param failedWorkerTaskArrayId
     * @throws InterruptedException
     * @throws KeeperException
     */
    public void weakupBackup(TensorflowTask backupWorkerTask, Integer failedWorkerTaskArrayId)
            throws KeeperException, InterruptedException {
        TensorflowTask[] workers = this.jobNameToTasks.get(Constants.WORKER_JOB_NAME);
        TensorflowTask failedWorkerTask = workers[failedWorkerTaskArrayId];
        backupWorkerTask.setTrainingDataPaths(failedWorkerTask.getTrainingDataPaths());
        backupWorkerTask.setArrayIndex(failedWorkerTaskArrayId);

        // write data path into zookeeper so that to weak up backup task
        LOG.info("failedWorkerTask.getTrainingDataPaths(): " + failedWorkerTask.getTrainingDataPaths());
        zookeeperServer.createOrSetExt(
                Constants.getTrainingDataZookeeperPath(backupWorkerTask.getContainer().getId().toString()),
                failedWorkerTask.getTrainingDataPaths().getBytes(Charset.forName("UTF-8")), Ids.OPEN_ACL_UNSAFE,
                CreateMode.PERSISTENT, true, -1);

        workers[failedWorkerTaskArrayId] = backupWorkerTask;
    }

    /**
     * Weak up backup worker with giving training data
     * 
     * @param backupWorkerTask
     * @param trainingDataPath
     * @throws InterruptedException
     * @throws KeeperException
     */
    public void weakupBackup(TensorflowTask backupWorkerTask, String trainingDataPath) {
        if(StringUtils.isBlank(trainingDataPath)) {
            return;
        }

        try {
            zookeeperServer.createOrSetExt(
                    Constants.getTrainingDataZookeeperPath(backupWorkerTask.getContainer().getId().toString()),
                    trainingDataPath.getBytes(Charset.forName("UTF-8")), Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT,
                    true, -1);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return the globalEpoch
     */
    public AtomicInteger getGlobalEpoch() {
        return globalEpoch;
    }

    /**
     * @param globalEpoch
     *            the globalEpoch to set
     */
    public void setGlobalEpoch(AtomicInteger globalEpoch) {
        this.globalEpoch = globalEpoch;
    }

    /**
     * @return the totalEpochs
     */
    public int getTotalEpochs() {
        return totalEpochs;
    }

    /**
     * @param totalEpochs
     *            the totalEpochs to set
     */
    public void setTotalEpochs(int totalEpochs) {
        this.totalEpochs = totalEpochs;
    }

    public TensorflowClusterSpec getTensorflowClusterSpec() {
        return tensorflowClusterSpec;
    }

    public static GuaguaZooKeeper getZookeeperServer() {
        return zookeeperServer;
    }

    public long getStartTimeOfRegisteringCluster() {
        return startTimeOfRegisteringCluster;
    }

    public void setStartTimeOfRegisteringCluster(long startTimeOfRegisteringCluster) {
        this.startTimeOfRegisteringCluster = startTimeOfRegisteringCluster;
    }

    public SessionState getState() {
        return state;
    }

    public void setState(SessionState state) {
        this.state = state;
    }

    public enum SessionState {
        INIT, STARTING_CONTAINER, REGESTERING_CLUSTER, TRAINING, FINISH
    }
}
