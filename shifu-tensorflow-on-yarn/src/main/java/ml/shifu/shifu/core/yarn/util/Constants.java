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

import java.io.File;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.yarn.api.ApplicationConstants;

import ml.shifu.shifu.util.HDFSUtils;

/**
 * @author webai
 *
 */
public class Constants {
    // Configuration related constants
    // Name of the file containing all configuration keys and their default values
    public static final String GLOBAL_DEFAULT_XML = "global-default.xml";
    // Default file name of user-provided configuration file
    public static final String GLOBAL_XML = "global.xml";
    // global-internal file name for final configurations, after user-provided configuration
    // file and CLI confs are combined. This file is uploaded to HDFS and localized to containers
    public static final String GLOBAL_FINAL_XML = "global-final.xml";
    public static final String JAR_LIB_ROOT = "lib";
    public static final String JAR_LIB_ZIP = "lib.zip";
    public static final String BACKUP_SCRIPT = "/backup.py";
    
    public static final String AM_NAME = "am";
    public static final String WORKER_JOB_NAME = "worker";
    public static final String PS_JOB_NAME = "ps";
    
    public static final String YARN_TMP = "tmp";
    public static final String SHIFU_TENSORFLOW_HDFS_DIR = "shifu_tensorflow";
    
    public static final String PYTHON_VENV_ZIP = "python_env.zip";
    public static final String GLIBC_VENV_ZIP = "glibc.zip";
    public static final String SHIFU_HISTORY_INTERMEDIATE = "intermediate";
    
    // History Server related constants
    public static final String JOBS_SUFFIX = "jobs";
    public static final String CONFIG_SUFFIX = "config";
    
    // File Permission
    public static final FsPermission PERM770 = new FsPermission((short) 0770);
    public static final FsPermission PERM777 = new FsPermission((short) 0777);
    
    // Environment variables for resource localization
    public static final String SHIFU_CONF_PREFIX = "SHIFU_CONF";
    public static final String PATH_SUFFIX = "_PATH";
    public static final String TIMESTAMP_SUFFIX = "_TIMESTAMP";
    public static final String LENGTH_SUFFIX = "_LENGTH";
    
    public static final String AM_STDOUT_FILENAME = "amstdout.log";
    public static final String AM_STDERR_FILENAME = "amstderr.log";
    
    // Zookeeper constants
    public static final String TENSORFLOW_CLUSTER_ROOT_PATH = "/tensorflow_cluster/";
    public static final String TENSORFLOW_FINAL_CLUSTER = "/tensorflow_cluster/final";
    public static final String BACKUP_WEKAINGUP_FLAG_PREFIX = "/backup_training_data_path_"; // it will conbime with container_id
    public static final String WORKER_INTERMEDIATE_RESULT_ROOT_PATH = "/worker_intermediate_result/";
    
    public static final String getTrainingDataZookeeperPath(String containerId) {
        return BACKUP_WEKAINGUP_FLAG_PREFIX + containerId;
    }
    
    public static final String ATTEMPT_NUMBER = "ATTEMPT_NUMBER";
    public static final String HADOOP_CONF_DIR = ApplicationConstants.Environment.HADOOP_CONF_DIR.key();;

    public static final String CORE_SITE_CONF = "core-site.xml";
    
    /** if failed workers number smaller than workerFaultToleranceThreashold, we still consider session success **/
    public static final double WORKER_FAULT_TOLERENANCE_THRESHOLD = 0.1d;
    public static final double PS_FAULT_TOLERNANCE_THREAHOLD = 0.9d;
    
    /** we will wait six minute, after that, we will do as best we can **/
    public static final int TIMEOUT_WAITING_CLUSTER_REGISTER = 6 * 60 * 1000;
    public static final double MIN_WORKERS_START_TRAINING_THREASHOLD = 0.95;
    public static final double MIN_PS_START_TRAINING_THREASHOLD = 0.95;
    
    public static final FileSystem hdfs = HDFSUtils.getFS();
    public static final Path getAppResourcePath(String appId) {
        return new Path(new Path(File.separator + YARN_TMP, SHIFU_TENSORFLOW_HDFS_DIR), appId);
    }
    
    public static String getClientResourcesPath(String appId, String fileName) {
        return String.format("%s-%s", appId, fileName);
    }
    
    public static String getProgressLogFile() {
        return new Path(new Path(File.separator + YARN_TMP, SHIFU_TENSORFLOW_HDFS_DIR), 
                String.format("%s.log", System.currentTimeMillis())).toString();
    }
}
