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

import java.util.Objects;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.yarn.api.records.Container;

import ml.shifu.shifu.core.yarn.container.TensorflowTaskExecutor;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;

/**
 * @author webai
 *
 */
public class TensorflowTask {
    private final String jobName;
    private final String taskIndex;
    private final int sessionId;
    /** The container the task is running in. Set once a container has been allocated for the task. */
    private Container container;
    private String trainingDataPaths; // data paths splited by ","
    
    private String zookeeperServer;
    private boolean isBackup;
    /** Task index in task array **/
    private int arrayIndex;
    
    private Configuration globalConf;

    public TensorflowTask(String jobName, String taskIndex, int sessionId, Container container,
            String zookeeperServer, Configuration globalConf, boolean isBackup, int arrayIndex) {
        this.jobName = jobName;
        this.taskIndex = taskIndex;
        this.sessionId = sessionId;
        this.container = container;
        this.zookeeperServer = zookeeperServer;
        this.globalConf = globalConf;
        this.isBackup = isBackup;
        this.arrayIndex = arrayIndex;
    }

    /** we get port only when job is executing, we will read it from zookeeper **/
    //private String tensorflowPort = null;

    int exitStatus = -1;

    /** Set to true when exit status is set. **/
    boolean completed = false;

    public String getJobName() {
        return jobName;
    }

    public int getSessionId() {
        return sessionId;
    }

    public String getTaskIndex() {
        return taskIndex;
    }

    public Container getContainer() {
        return container;
    }

    public void setContainer(Container container) {
        this.container = container;
    }

    public boolean isCompleted() {
        return completed;
    }

    public int getExitStatus() {
        return exitStatus;
    }

    public String getHostName() {
        return this.container.getNodeId().getHost();
    }

    public int getArrayIndex() {
        return arrayIndex;
    }
    
    public void setArrayIndex(int arrayIndex) {
        this.arrayIndex = arrayIndex;
    }
    
    public String getTrainingDataPaths() {
        return trainingDataPaths;
    }

    public void setTrainingDataPaths(String trainingDataPaths) {
        this.trainingDataPaths = trainingDataPaths;
    }
//    public String getHostNameAndPort() {
//        return String.format("%s:%s", getHostName(), StringUtils.isBlank(tensorflowPort) ? 0 : tensorflowPort);
//    }
//
//    public void setTensorflowPort(String port) {
//        tensorflowPort = port;
//    }

    void setExitStatus(int status) {
        this.completed = true;
        this.exitStatus = status;
    }

    /**
     * Returns a {@link TaskUrl} containing the HTTP URL for the task.
     */
    public TaskUrl getTaskUrl() {
        if(container == null) {
            return null;
        }
        return new TaskUrl(jobName, taskIndex, CommonUtils.constructContainerUrl(container));
    }

    public String getTaskCommand() {
        StringBuilder cmd = new StringBuilder();
        cmd.append("$JAVA_HOME/bin/java ")
                .append(globalConf.get(GlobalConfigurationKeys.TASK_EXECUTOR_JVM_OPTS, GlobalConfigurationKeys.DEFAULT_TASK_EXECUTOR_JVM_OPTS))
                .append(" " + TensorflowTaskExecutor.class.getName() + " ")
                .append(" --zookeeper_server ").append(zookeeperServer)
                .append(" --job_name ").append(jobName)
                .append(" --task_id ").append(taskIndex)
                .append(" --container_id ").append(container.getId().toString())
                .append(" --is_backup ").append(isBackup);
        if (StringUtils.isNotBlank(trainingDataPaths)){
            cmd.append(" --training_data_path ").append(trainingDataPaths);
        }
        return cmd.toString();
    }

    /**
     * Combination of jobName and taskIndex.
     * 
     * @return Id
     */
    public String getId() {
        return this.jobName + ":" + this.taskIndex;
    }

    @Override
    public boolean equals(Object o) {
        if(this == o) {
            return true;
        }
        if(o == null || getClass() != o.getClass()) {
            return false;
        }
        TensorflowTask TensorflowTask = (TensorflowTask) o;
        return Objects.equals(jobName, TensorflowTask.jobName) && Objects.equals(taskIndex, TensorflowTask.taskIndex);
    }

    @Override
    public int hashCode() {
        return Objects.hash(jobName, taskIndex);
    }
}
