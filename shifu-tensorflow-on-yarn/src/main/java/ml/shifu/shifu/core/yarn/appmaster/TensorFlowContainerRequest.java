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

/**
 * @author webai
 *
 */
public class TensorFlowContainerRequest {
    private int numInstances;
    private int numBackupInstances;
    private long memory;
    private int vCores;
    private int priority;
    private String jobName;

    public TensorFlowContainerRequest(String jobName, int numInstances, long memory, int vCores, int priority, int numBackupInstances) {
      this.numInstances = numInstances;
      this.numBackupInstances = numBackupInstances;
      this.memory = memory;
      this.vCores = vCores;
      this.priority = priority;
      this.jobName = jobName;
    }

    public TensorFlowContainerRequest(TensorFlowContainerRequest that) {
      this.numInstances = that.numInstances;
      this.memory = that.memory;
      this.vCores = that.vCores;
      this.priority = that.priority;
      this.jobName = that.jobName;
      this.numBackupInstances = that.numBackupInstances;
    }

    public int getNumInstances() {
      return numInstances;
    }

    public long getMemory() {
      return memory;
    }

    public int getVCores() {
      return vCores;
    }

    public int getPriority() {
      return priority;
    }

    public String getJobName() {
      return jobName;
    }
    public int getNumBackupInstances() {
        return numBackupInstances;
    }
}
