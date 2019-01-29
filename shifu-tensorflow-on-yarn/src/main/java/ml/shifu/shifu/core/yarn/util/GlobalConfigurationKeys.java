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
package ml.shifu.shifu.core.yarn.util;

/**
 * @author webai
 *
 */
public class GlobalConfigurationKeys {
    private GlobalConfigurationKeys() {}
    
    public static final String SHIFU_PREFIX = "shifu.";
    
    // Application Configuration
    public static final String SHIFU_APPLICATION_PREFIX = SHIFU_PREFIX + "application.";
    
    public static final String APPLICATION_TIMEOUT = SHIFU_APPLICATION_PREFIX + "timeout";
    public static final int DEFAULT_APPLICATION_TIMEOUT = 0;
    
    public static final String RM_CLIENT_CONNECT_RETRY_MULTIPLIER = SHIFU_APPLICATION_PREFIX + "num-client-rm-connect-retries";
    public static final int DEFAULT_RM_CLIENT_CONNECT_RETRY_MULTIPLIER = 3;
    
    public static final String APPLICATION_NAME = SHIFU_APPLICATION_PREFIX + "name";
    public static final String DEFAULT_APPLICATION_NAME = "ShifuTensorflowApplication";
    
    // Application configurations
    public static final String YARN_QUEUE_NAME = SHIFU_APPLICATION_PREFIX + "yarn.queue";
    public static final String DEFAULT_YARN_QUEUE_NAME = "default";
    
    // History folder configuration
    public static final String SHIFU_HISTORY_HOST = SHIFU_PREFIX + "history.host";
    public static final String DEFAULT_SHIFU_HISTORY_HOST = "historyhost.com";
    
    public static final String SHIFU_HISTORY_LOCATION = SHIFU_PREFIX + "history.location";
    public static final String DEFAULT_SHIFU_HISTORY_LOCATION = "/path/to/shifu-history";
    
    //AM Configuration
    public static final String AM_PREFIX = SHIFU_PREFIX + "am.";
    
    public static final String AM_MEMORY = AM_PREFIX + "memory";
    public static final String DEFAULT_AM_MEMORY = "2g";
    
    public static final String AM_VCORES = AM_PREFIX + "vcores";
    public static final int DEFAULT_AM_VCORES = 1;
    
    public static final String AM_RETRY_COUNT = AM_PREFIX + "retry-count";
    public static final int DEFAULT_AM_RETRY_COUNT = 0;
    
    // Worker configurations
    public static final String WORKER_PREFIX = SHIFU_PREFIX + "worker.";
    public static final String WORKER_TIMEOUT = WORKER_PREFIX + "timeout";
    public static final int DEFAULT_WORKER_TIMEOUT = 0;
    
    // Task configurations
    public static final String SHIFU_TASK_PREFIX = SHIFU_PREFIX + "task.";

    public static final String TASK_EXECUTOR_JVM_OPTS = SHIFU_TASK_PREFIX + "executor.jvm.opts";
    public static final String DEFAULT_TASK_EXECUTOR_JVM_OPTS = "-Xmx1536m";

    public static final String TASK_HEARTBEAT_INTERVAL_MS = SHIFU_TASK_PREFIX + "heartbeat-interval";
    public static final int DEFAULT_TASK_HEARTBEAT_INTERVAL_MS = 1000;

    public static final String TASK_MAX_MISSED_HEARTBEATS = SHIFU_TASK_PREFIX + "max-missed-heartbeats";
    public static final int DEFAULT_TASK_MAX_MISSED_HEARTBEATS = 25;
    
    public static final String HDFS_CONF_LOCATION = SHIFU_APPLICATION_PREFIX + "hdfs-conf-path";

    public static final String YARN_CONF_LOCATION = SHIFU_APPLICATION_PREFIX + "yarn-conf-path";

    public static final String TRAINING_DATA_PATH = SHIFU_APPLICATION_PREFIX + "training-data-path";
    public static final String WEIGHT_COLUMN_NUM = SHIFU_APPLICATION_PREFIX + "weight-column-number";
    public static final String DEFAULT_WEIGHT_COLUMN_NUM = "-1";
    
    public static final String PYTHON_BINARY_PATH = SHIFU_APPLICATION_PREFIX + "python-binary-path";
    public static final String GLIBC_BINARY_PATH = SHIFU_APPLICATION_PREFIX + "glibc-binary-path";
    public static final String PYTHON_SCRIPT_PATH = SHIFU_APPLICATION_PREFIX + "python-script-path";
    public static final String SHIFU_YARN_APP_JAR = SHIFU_APPLICATION_PREFIX + "app-jar-path";

    // Keys/default values for configurable TensorFlow job names
    public static final String INSTANCES_REGEX = "shifu\\.([a-z]+)\\.instances";
    public static final String DEFAULT_MEMORY = "2g";
    public static final int DEFAULT_VCORES = 1;
    public static final int DEFAULT_GPUS = 0;
    
    // Resources for all containers in hdfs
    public static String getContainerResourcesKey() {
      return SHIFU_PREFIX + "containers.resources";
    }
    
    public static String getInstancesKey(String jobName) {
        return String.format(SHIFU_PREFIX + "%s.instances", jobName);
    }
    
    public static int getDefaultInstances(String jobName) {
        if (Constants.WORKER_JOB_NAME.equalsIgnoreCase(jobName)) {
            return 1;
        } else {
            return 0;
        }
    }
    
    // Job specific resources
    public static String getResourcesKey(String jobName) {
        return String.format(SHIFU_PREFIX + "%s.resources", jobName);
    }

    public static String getMemoryKey(String jobName) {
        return String.format(SHIFU_PREFIX + "%s.memory", jobName);
    }

    public static String getVCoresKey(String jobName) {
        return String.format(SHIFU_PREFIX + "%s.vcores", jobName);
    }
    
    public static String getBackupInstancesKey(String jobName) {
        return String.format(SHIFU_PREFIX + "%s.instances.backup", jobName);
    }
}
