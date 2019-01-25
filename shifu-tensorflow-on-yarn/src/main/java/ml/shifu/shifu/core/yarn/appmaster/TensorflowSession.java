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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
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
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.ZooDefs.Ids;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.google.common.base.Preconditions;

import ml.shifu.guagua.coordinator.zk.GuaguaZooKeeper;
import ml.shifu.guagua.coordinator.zk.ZooKeeperUtils;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.Constants;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;

/**
 * @author webai
 * 
 *         Tensorflow Session contains all tensorflow jobs information
 */
public class TensorflowSession implements Watcher {
    private static final Log LOG = LogFactory.getLog(TensorflowSession.class);
    private Configuration globalConf;

    private Map<String, TensorFlowContainerRequest> containerRequests;
    private Map<String, List<ContainerRequest>> jobNameToContainerRequests = new HashMap<String, List<ContainerRequest>>();
    private Map<String, TensorflowTask> containerIdToTask = new HashMap<String, TensorflowTask>();
    // A map from task name to an array of TFTasks with that name.
    private Map<String, TensorflowTask[]> jobNameToTasks = new ConcurrentHashMap<String, TensorflowTask[]>();
    private TensorflowClusterSpec tensorflowClusterSpec;

    /** those task not have container **/
    private Map<String, Integer> jobNameToPendingTaskNumber = new ConcurrentHashMap<String, Integer>();
    private int numRequestedContainers = 0;

    /** train data set **/
    private List<StringBuilder> splitedTrainingData = null;

    /** Job progress **/
    private AtomicInteger numCompletedWorkerTasks = new AtomicInteger(0);
    private int numTotalWorkerTasks = 1;
    private int numTotalPsTasks = 1;

    /** if failed workers number smaller than workerFaultToleranceThreashold, we still consider session success **/
    private static final double workerFaultToleranceThreashold = 0.1d;

    /** failed including timeout task and return wrong exit code **/
    private AtomicInteger failedWorkers = new AtomicInteger(0);
    /** failed only when return wrong exit code **/
    private AtomicInteger failedPs = new AtomicInteger(0);

    // sessionId to distinguish different sessions. Currently used to distinguish
    // failed session and new session.
    public static int sessionId = 0;

    private FinalApplicationStatus sessionFinalStatus = FinalApplicationStatus.UNDEFINED;
    private String sessionFinalMessage = null;

    // if Chief worker finished with non-zero exit code, we stop whole training
    private boolean chiefWorkerSuccess = true;

    private static String zookeeperServerHostPort = null;
    private static GuaguaZooKeeper zookeeperServer = null;

    public enum TaskType {
        TASK_TYPE_CHIEF, TASK_TYPE_PARAMETER_SERVER, TASK_TYPE_WORKER
    }
    public TensorflowSession() {}
    public TensorflowSession(Configuration globalConf) {
        this.globalConf = globalConf;
        this.containerRequests = CommonUtils.parseContainerRequests(this.globalConf);

        // create zookeeper server for sync tensorflow cluster spec
        // This has been settled in prepare of AM
        if (zookeeperServer == null) {
            zookeeperServerHostPort = startZookeeperServer();
            try {
                this.zookeeperServer = new GuaguaZooKeeper(zookeeperServerHostPort, 300000, 5, 1000, this);
            } catch (IOException e) {
                LOG.error("create zookeeper server fails!", e);
                throw new RuntimeException(e);
            }
        }

        for(String jobName: containerRequests.keySet()) {
            int taskCnt = containerRequests.get(jobName).getNumInstances();

            jobNameToTasks.put(jobName, new TensorflowTask[taskCnt]);

            // set tasks number in order to do auditing
            if(Constants.WORKER_JOB_NAME.equals(jobName)) {
                this.numTotalWorkerTasks = taskCnt;
            } else if(Constants.PS_JOB_NAME.equals(jobName)) {
                this.numTotalPsTasks = taskCnt;
            }
        }

        this.tensorflowClusterSpec = new TensorflowClusterSpec(numTotalPsTasks, numTotalWorkerTasks);
        
        // Split training data for workers
        try {
            splitedTrainingData = TrainingDataSet.getInstance().getSplitedFilePaths(this.globalConf, this.numTotalWorkerTasks,
                    this.globalConf.get(GlobalConfigurationKeys.TRAINING_DATA_PATH));
        } catch (Exception e) {
            LOG.error("Splitting training data fails!", e);
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
        for(String jobName: containerRequests.keySet()) {
            if(!jobNameToContainerRequests.containsKey(jobName)) {
                jobNameToContainerRequests.put(jobName, new ArrayList<ContainerRequest>());
            }

            TensorFlowContainerRequest containerRequest = containerRequests.get(jobName);
            // prepare resource request and add to amRMClient
            for(int i = 0; i < containerRequest.getNumInstances(); i++) {
                AMRMClient.ContainerRequest containerAsk = setupContainerRequestForRM(containerRequest);

                jobNameToContainerRequests.get(jobName).add(containerAsk);

                
                amRMClient.addContainerRequest(containerAsk);

                numRequestedContainers++;
            }

            jobNameToPendingTaskNumber.put(jobName, containerRequest.getNumInstances());
        }
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
            return null;
            //throw new RuntimeException("couldn't find job to match container");
        }

        TensorflowTask[] tasks = jobNameToTasks.get(jobName);
        for(int i = 0; i < tasks.length; i++) {
            if(tasks[i] == null) {
                try {
                    zookeeperServer.exists(Constants.TENSORFLOW_CLUSTER_ROOT_PATH + container.getId().toString(), this);
                } catch (Exception e) {
                    LOG.error("watch container fails", e);
                    throw new RuntimeException(e);
                }
                
                tasks[i] = new TensorflowTask(jobName, String.valueOf(i), sessionId, container,
                        this.splitedTrainingData.get(i).toString(), zookeeperServerHostPort, this.globalConf);

                jobNameToPendingTaskNumber.put(jobName, jobNameToPendingTaskNumber.get(jobName) - 1);
                containerIdToTask.put(container.getId().toString(), tasks[i]);
                return tasks[i];
            }
        }

        return null;
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
        LOG.info("remaining resource: " +jobNameToPendingTaskNumber);
        LOG.info("session id: " + sessionId);
        for(Map.Entry<String, TensorFlowContainerRequest> jobNameToRequest: containerRequests.entrySet()) {
            String jobName = jobNameToRequest.getKey();
            TensorFlowContainerRequest request = jobNameToRequest.getValue();
            int pendingNumber = jobNameToPendingTaskNumber.get(jobName);

            if((int) request.getMemory() == container.getResource().getMemory()
                    && request.getVCores() == container.getResource().getVirtualCores() && pendingNumber > 0) {
                return jobName;
            }
        }

        return null;
    }

    public TensorflowTask getTaskByContainerId(ContainerId containerId) {
        return this.containerIdToTask.get(containerId.toString());
    }

    public TensorflowTask getTaskByJobnameAndTaskId(String jobName, String taskIndex) {
        for(Map.Entry<String, TensorflowTask[]> entry: this.jobNameToTasks.entrySet()) {
            TensorflowTask[] tasks = entry.getValue();
            for(TensorflowTask task: tasks) {
                String job = task.getJobName();
                String index = task.getTaskIndex();
                if(job.equals(jobName) && index.equals(taskIndex)) {
                    return task;
                }
            }
        }
        return null;
    }

    private TaskType getTaskType(TensorflowTask task) {
        TaskType type;
        String jobName = task.getJobName();
        if(jobName.equals(Constants.PS_JOB_NAME)) {
            type = TaskType.TASK_TYPE_PARAMETER_SERVER;
        } else {
            type = TaskType.TASK_TYPE_WORKER;
        }
        return type;
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

    /**
     * Update the status of a session and set exit code if a session is completed.
     */
    public void updateSessionStatus() {
        if(getFinalStatus() == FinalApplicationStatus.FAILED) {
            return;
        }
        for(Map.Entry<String, TensorflowTask[]> entry: this.jobNameToTasks.entrySet()) {
            TensorflowTask[] tasks = entry.getValue();
            for(TensorflowTask task: tasks) {
                if(task == null) {
                    String msg = "Job is null, this should not happen.";
                    LOG.error(msg);
                    setFinalStatus(FinalApplicationStatus.FAILED, msg);
                    return;
                }
                boolean isCompleted = task.isCompleted();
                if(!isCompleted) {
                    if(Constants.WORKER_JOB_NAME.equals(task.getJobName())) {
                        this.failedWorkers.incrementAndGet();
                    }

                    String msg = "Job " + task.getJobName() + " at index: " + task.getTaskIndex()
                            + " haven't finished yet.";
                    LOG.error(msg);
                }
            }

        }

        if(this.failedPs.get() >= this.numTotalPsTasks) {
            setFinalStatus(FinalApplicationStatus.FAILED, "There is no PS sucess, failedCnt=" + this.failedPs.get());
        } else if(this.failedWorkers.get() >= (this.numTotalWorkerTasks * workerFaultToleranceThreashold)) {
            setFinalStatus(FinalApplicationStatus.FAILED,
                    "More than 10% of worker failed, failedCnt=" + this.failedWorkers.get());
        } else if(!chiefWorkerSuccess) {
            setFinalStatus(FinalApplicationStatus.FAILED, "Chief worker failed");
        } else {
            LOG.info("Session completed with no job failures, setting final status SUCCEEDED.");
            setFinalStatus(FinalApplicationStatus.SUCCEEDED, null);
        }
    }

    /**
     * Refresh task status on each TaskExecutor registers its exit code with AM.
     */
    public void onTaskCompleted(String jobName, String jobIndex, int exitCode) {
        LOG.info(String.format("Job %s:%s exited with %d", jobName, jobIndex, exitCode));
        TensorflowTask task = getTaskByJobnameAndTaskId(jobName, jobIndex);
        Preconditions.checkNotNull(task);
        TaskType taskType = getTaskType(task);
        task.setExitStatus(exitCode);
        switch(taskType) {
            case TASK_TYPE_CHIEF:
            case TASK_TYPE_PARAMETER_SERVER:
                if(exitCode != 0) {
                    this.failedPs.incrementAndGet();
                }
                break;
            case TASK_TYPE_WORKER:
                // If the chief worker failed[chief or worker 0], short circuit and stop the training. Note that even
                // though other worker failures will also fail the job but we don't short circuit the training because
                // the training
                // can still continue, while if chief worker is dead, a TensorFlow training would hang.
                // Also note that, we only short circuit when the chief worker failed, not finished.
                if(exitCode != 0) {
                    if(isChief(jobName, jobIndex)) {
                        chiefWorkerSuccess = false;
                    }
                    this.failedWorkers.incrementAndGet();
                }
                break;
            default:
                break;
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

    public AtomicInteger getNumCompletedWorkerTasks() {
        return numCompletedWorkerTasks;
    }

    public void setNumCompletedWorkerTasks(AtomicInteger numCompletedWorkerTasks) {
        this.numCompletedWorkerTasks = numCompletedWorkerTasks;
    }

    public int getNumTotalWorkerTasks() {
        return numTotalWorkerTasks;
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.zookeeper.Watcher#process(org.apache.zookeeper.WatchedEvent)
     */
    public void process(WatchedEvent event) {
        if (event == null || StringUtils.isBlank(event.getPath())) {
            return;
        }
        
        String containerId = event.getPath().replace(Constants.TENSORFLOW_CLUSTER_ROOT_PATH, "");
        try {
            String ipAndPort = new String(zookeeperServer.getData(event.getPath(), null, null));
            TensorflowTask task = this.containerIdToTask.get(containerId);
            this.tensorflowClusterSpec.add(task.getJobName(), Integer.valueOf(task.getTaskIndex()), ipAndPort);
        } catch (Exception e) {
            LOG.error("Getting worker port fails.", e);
            throw new RuntimeException(e);
        }

        int readyContainersNumber = this.tensorflowClusterSpec.totalWorkerAndPs();
        if(readyContainersNumber == numRequestedContainers) {
            LOG.info("Get all host and port from containers");
            try {
                this.zookeeperServer.createOrSetExt(Constants.TENSORFLOW_FINAL_CLUSTER,
                        tensorflowClusterSpec.toString().getBytes(Charset.forName("UTF-8")), Ids.OPEN_ACL_UNSAFE,
                        CreateMode.PERSISTENT, true, -1);
            } catch (Exception e) {
                LOG.fatal("Writing final cluster spec to zookeeper fails.", e);
                throw new RuntimeException(e);
            }
        } else if(readyContainersNumber < numRequestedContainers) {
            LOG.info("total: " + numRequestedContainers + ", ready: " + readyContainersNumber);
        } else {
            LOG.fatal("total: " + numRequestedContainers + ", ready: " + readyContainersNumber);
        }
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
    
    class TensorflowClusterSpec {
        // In order to make spec host order same as task order
        private String[] ps;
        private String[] worker;
        private int readyPsCnt = 0;
        private int readyWorkerCnt = 0;
        
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
        }

        public String[] getPs() {
            return ps;
        }
        
        public String[] getWorker() {
            return worker;
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
            //return ToStringBuilder.reflectionToString(this, ToStringStyle.JSON_STYLE);
            return StringUtils.EMPTY;
        }
    }
}
