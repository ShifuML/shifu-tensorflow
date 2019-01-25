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
import java.io.IOException;
import java.net.InetAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.records.ApplicationAttemptId;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.ConverterUtils;

import ml.shifu.guagua.GuaguaRuntimeException;
import ml.shifu.shifu.core.yarn.appmaster.TaskUrl;
import ml.shifu.shifu.core.yarn.appmaster.TensorFlowContainerRequest;
import ml.shifu.shifu.core.yarn.client.TensorflowClient;
import ml.shifu.shifu.util.HDFSUtils;
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;

/**
 * @author webai
 *
 */
public class CommonUtils {
    private static final Log LOG = LogFactory.getLog(CommonUtils.class);

    /**
     * This function is used by TensorflowApplicationMaster and TensorflowClient to set up
     * common command line arguments.
     * 
     * @return Options that contains common options
     */
    public static Options getCommonOptions() {
        Options opts = new Options();

        // Execution
        opts.addOption("task_params", true, "The task params to pass into python entry point.");
        opts.addOption("executes", true, "The file to execute on workers.");

        // Python env
        opts.addOption("python_binary_path", true, "The relative path to python binary.");
        opts.addOption("python_venv", true, "The python virtual environment zip.");

        // Glibc env
        opts.addOption("glibc_venv", true, "The glibc zip file.");
        opts.addOption("glibc_binary_path", true, "The relative path to glibc lib");

        // Container environment
        // examples for env set variables: --shell_env CLASSPATH=ABC --shell_ENV LD_LIBRARY_PATH=DEF
        opts.addOption("container_env", true, "Environment for the worker containers, specified as key=val pairs");
        // opts.addOption("shell_env", true, "Environment for shell script, specified as env_key=env_val pairs");
        // opts.addOption("hdfs_classpath", true, "Path to jars on HDFS for workers.");

        return opts;
    }

    /**
     * Parse a list of env key-value pairs like PATH=ABC to a map of key value entries.
     * 
     * @param keyValues
     *            the input key value pairs
     * @return a map contains the key value {"PATH": "ABC"}
     */
    public static Map<String, String> parseKeyValue(String[] keyValues) {
        Map<String, String> keyValue = new HashMap<String, String>();
        if(keyValues == null) {
            return keyValue;
        }
        for(String kv: keyValues) {
            String trimmedKeyValue = kv.trim();
            int index = kv.indexOf('=');
            if(index == -1) {
                keyValue.put(trimmedKeyValue, "");
                continue;
            }
            String key = trimmedKeyValue.substring(0, index);
            String val = "";
            if(index < (trimmedKeyValue.length() - 1)) {
                val = trimmedKeyValue.substring(index + 1);
            }
            keyValue.put(key, val);
        }
        return keyValue;
    }

    public static String parseMemoryString(String memory) {
        memory = memory.toLowerCase();
        int m = memory.indexOf('m');
        int g = memory.indexOf('g');
        if(-1 != m) {
            return memory.substring(0, m);
        }
        if(-1 != g) {
            return String.valueOf(Integer.parseInt(memory.substring(0, g)) * 1024);
        }
        return memory;
    }

    public static void printTHSUrl(String thsHost, String appId, Log log) {
        log.info(String.format("Link for %s's events/metrics: http://%s/%s/%s", appId, thsHost, Constants.JOBS_SUFFIX,
                appId));
    }

    /**
     * Add files inside a path to local resources. If the path is a directory, its first level files will be added
     * to the local resources. Note that we don't add nested files.
     * 
     * @param path
     *            the directory whose contents will be localized.
     * @param fs
     *            the configuration file for HDFS.
     */
    public static void addResource(String path, Map<String, LocalResource> resourcesMap, FileSystem fs) {
        try {
            if(path != null) {
                FileStatus[] ls = fs.listStatus(new Path(path));
                for(FileStatus jar: ls) {
                    // We only add first level files.
                    if(jar.isDirectory()) {
                        continue;
                    }
                    LocalResource resource = LocalResource.newInstance(
                            ConverterUtils.getYarnUrlFromURI(URI.create(jar.getPath().toString())),
                            LocalResourceType.FILE, LocalResourceVisibility.APPLICATION, jar.getLen(),
                            jar.getModificationTime());
                    resourcesMap.put(jar.getPath().getName(), resource);
                }
            }
        } catch (IOException exception) {
            LOG.error("Failed to add " + path + " to local resources.", exception);
        }
    }

    public static void addResource(Path path, Map<String, LocalResource> resourcesMap, FileSystem hdfs,
            LocalResourceType resourceType, String resourceKey) {
        FileStatus scFileStatus;
        try {
            scFileStatus = hdfs.getFileStatus(path);
            LocalResource scRsrc = LocalResource.newInstance(ConverterUtils.getYarnUrlFromURI(path.toUri()),
                    resourceType, LocalResourceVisibility.APPLICATION, scFileStatus.getLen(),
                    scFileStatus.getModificationTime());
            resourcesMap.put(resourceKey, scRsrc);
        } catch (IOException e) {
            LOG.error("Failed to add " + path + " to local resources.", e);
        }

    }

    public static void addEnvironmentForResource(LocalResource resource, FileSystem fs, String envPrefix,
            Map<String, String> env) throws IOException {
        Path resourcePath = new Path(fs.getHomeDirectory(), resource.getResource().getFile());
        FileStatus resourceStatus = fs.getFileStatus(resourcePath);
        long resourceLength = resourceStatus.getLen();
        long resourceTimestamp = resourceStatus.getModificationTime();

        env.put(envPrefix + Constants.PATH_SUFFIX, resourcePath.toString());
        env.put(envPrefix + Constants.LENGTH_SUFFIX, Long.toString(resourceLength));
        env.put(envPrefix + Constants.TIMESTAMP_SUFFIX, Long.toString(resourceTimestamp));
    }

    private static final String WORKER_LOG_URL_TEMPLATE = "http://%s/node/containerlogs/%s/%s";

    public static String constructContainerUrl(Container container) {
        try {
            return String.format(WORKER_LOG_URL_TEMPLATE, container.getNodeHttpAddress(), container.getId(),
                    UserGroupInformation.getCurrentUser().getShortUserName());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static String getCurrentHostName() {
        //return System.getenv(ApplicationConstants.Environment.NM_HOST.name());
        
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            throw new GuaguaRuntimeException(e);
        }
    }
    
    public static String getCurrentHostIP() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (Exception e) {
            throw new GuaguaRuntimeException(e);
        }
    }

    /**
     * Execute a shell command.
     * 
     * @param taskCommand
     *            the shell command to execute
     * @param timeout
     *            the timeout to stop running the shell command
     * @param env
     *            the environment for this shell command
     * @return the exit code of the shell command
     * @throws IOException
     * @throws InterruptedException
     */
    public static int executeShell(String taskCommand, long timeout, Map<String, String> env)
            throws IOException, InterruptedException {
        LOG.info("Executing command: " + taskCommand);
        String executablePath = taskCommand.trim().split(" ")[0];
        File executable = new File(executablePath);
        if(!executable.canExecute()) {
            executable.setExecutable(true);
        }

        ProcessBuilder taskProcessBuilder = new ProcessBuilder(taskCommand);
        taskProcessBuilder.redirectError(ProcessBuilder.Redirect.INHERIT);
        taskProcessBuilder.redirectOutput(ProcessBuilder.Redirect.INHERIT);
        
        if(env != null) {
            putAll(taskProcessBuilder.environment(), env);
        }
        Process taskProcess = taskProcessBuilder.start();
        if(timeout > 0) {
            taskProcess.waitFor(timeout, TimeUnit.MILLISECONDS);
        } else {
            taskProcess.waitFor();
        }

        return taskProcess.exitValue();

    }

    private static void putAll(Map<String, String> target, Map<String, String> newEnv) {
        for (Entry<String, String> cur : newEnv.entrySet()) {
            LOG.info(cur.getKey() + ":" + cur.getValue());
            target.put(cur.getKey(), cur.getValue());
        }
    }
    
    /**
     * Parses resource requests from configuration of the form "shifu.x(PS).y(memory)" where "x" is the
     * TensorFlow job name, and "y" is "instances" or the name of a resource type
     * (i.e. memory, vcores, gpus).
     * 
     * @param conf
     *            the global configuration.
     * @return map from configured job name to its corresponding resource request
     */
    public static Map<String, TensorFlowContainerRequest> parseContainerRequests(Configuration conf) {
        List<String> jobNames = new ArrayList<String>();
        jobNames.add("ps");
        jobNames.add("worker");

        Map<String, TensorFlowContainerRequest> containerRequests = new HashMap<String, TensorFlowContainerRequest>();
        int priority = 0;
        for(String jobName: jobNames) {
            int numInstances = conf.getInt(GlobalConfigurationKeys.getInstancesKey(jobName),
                    GlobalConfigurationKeys.getDefaultInstances(jobName));
            String memoryString = conf.get(GlobalConfigurationKeys.getMemoryKey(jobName),
                    GlobalConfigurationKeys.DEFAULT_MEMORY);
            long memory = Long.parseLong(parseMemoryString(memoryString));
            int vCores = conf.getInt(GlobalConfigurationKeys.getVCoresKey(jobName),
                    GlobalConfigurationKeys.DEFAULT_VCORES);
            /*
             * The priority of different task types MUST be different.
             * Otherwise the requests will overwrite each other on the RM
             * scheduling side. See YARN-7631 for details.
             * For now we set the priorities of different task types arbitrarily.
             */
            if(numInstances > 0) {
                containerRequests.put(jobName,
                        new TensorFlowContainerRequest(jobName, numInstances, memory, vCores, priority++));
            }
        }
        return containerRequests;
    }

    public static void printWorkerTasksCompleted(AtomicInteger completedWTasks, long totalWTasks) {
        if(completedWTasks.get() == totalWTasks) {
            LOG.info("Completed all " + totalWTasks + " worker tasks.");
            return;
        }
        LOG.info("Completed worker tasks: " + completedWTasks.get() + " out of " + totalWTasks + " worker tasks.");
    }

    public static void printTaskUrl(TaskUrl taskUrl, Log log) {
        log.info(String.format("Logs for %s %s at: %s", taskUrl.getName(), taskUrl.getIndex(), taskUrl.getUrl()));
    }

    public static String buildRMUrl(Configuration yarnConf, String appId) {
        return "http://" + yarnConf.get(YarnConfiguration.RM_WEBAPP_ADDRESS) + "/cluster/app/" + appId;
    }

    public static void unzipArchive(String src, String dst) {
        LOG.info("Unzipping " + src + " to destination " + dst);
        try {
            File dstFile = new File(dst);
            if (!dstFile.exists()) {
                dstFile.mkdirs();
            }
            
            ZipFile zipFile = new ZipFile(src);
            zipFile.extractAll(dst);
        } catch (ZipException e) {
            LOG.fatal("Failed to unzip " + src, e);
        }
    }
    
    
}
