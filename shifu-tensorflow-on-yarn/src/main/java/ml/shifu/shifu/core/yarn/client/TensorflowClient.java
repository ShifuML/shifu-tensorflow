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
package ml.shifu.shifu.core.yarn.client;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.protocolrecords.GetNewApplicationResponse;
import org.apache.hadoop.yarn.api.records.ApplicationAccessType;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;

import com.google.common.collect.ImmutableList;

import ml.shifu.shifu.core.yarn.appmaster.TensorflowApplicationMaster;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.Constants;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;
import ml.shifu.shifu.core.yarn.util.HdfsUtils;
import ml.shifu.shifu.util.HDFSUtils;

/**
 * @author webai
 *
 */
public class TensorflowClient implements AutoCloseable{
    private static final Log LOG = LogFactory.getLog(TensorflowClient.class);

    private YarnClient yarnClient;
    private int amMemory;
    private int amVCores;
    private int hbInterval;
    private int maxHbMisses;
    private int appTimeout;

    // TODO: currently, we only use specific training script. this will be configurable in future
    // private String executes;
    private String localPythonVenv = "/x/home/webai/tensorflow/python_env.zip";
    private String localGlibcVenv = "/x/home/webai/tensorflow/glibc.zip";
    private String localJarLibPath = "/x/home/webai/tensorflow/lib.zip";
    private String localAppJarPath = null;
    private String localGlobalFinalConfPath = null;
    
    // resource folder on hdfs
    private Path appResourcesPath;
    private Path globalResourcesPath;
    
    private final long clientStartTime = System.currentTimeMillis();

    // Configurations
    private Configuration globalConf;
    private YarnConfiguration yarnConf = new YarnConfiguration(); 
    
    public TensorflowClient(Configuration conf) {
        globalConf = conf;
    }

    public TensorflowClient() {
        this(new Configuration(false));
    }

    /**
     * Parse command line options
     * 
     * @param args
     *            Parsed command line options
     * @return Whether the init was successful to run the client
     * @throws org.apache.commons.cli.ParseException
     */
    public boolean init(String[] args) throws Exception {
        globalConf.addResource(Constants.GLOBAL_DEFAULT_XML);
        
        String amMemoryString = globalConf.get(GlobalConfigurationKeys.AM_MEMORY,
                GlobalConfigurationKeys.DEFAULT_AM_MEMORY);
        amMemory = Integer.parseInt(CommonUtils.parseMemoryString(amMemoryString));
        amVCores = globalConf.getInt(GlobalConfigurationKeys.AM_VCORES, GlobalConfigurationKeys.DEFAULT_AM_VCORES);
        hbInterval = globalConf.getInt(GlobalConfigurationKeys.TASK_HEARTBEAT_INTERVAL_MS,
                GlobalConfigurationKeys.DEFAULT_TASK_HEARTBEAT_INTERVAL_MS);
        maxHbMisses = globalConf.getInt(GlobalConfigurationKeys.TASK_MAX_MISSED_HEARTBEATS,
                GlobalConfigurationKeys.DEFAULT_TASK_MAX_MISSED_HEARTBEATS);

        LOG.info("heartbeat interval [" + hbInterval + "]");
        LOG.info("max heartbeat misses allowed [" + maxHbMisses + "]");

        if(amMemory < 0) {
            throw new IllegalArgumentException(
                    "Invalid memory specified for application master, exiting." + " Specified memory=" + amMemory);
        }
        if(amVCores < 0) {
            throw new IllegalArgumentException("Invalid virtual cores specified for application master, exiting."
                    + " Specified virtual cores=" + amVCores);
        }

        int numWorkers = globalConf.getInt(GlobalConfigurationKeys.getInstancesKey(Constants.WORKER_JOB_NAME),
                GlobalConfigurationKeys.getDefaultInstances(Constants.WORKER_JOB_NAME));

        if(numWorkers < 1) {
            throw new IllegalArgumentException(
                    "Cannot request non-positive worker instances. Requested numWorkers=" + numWorkers);
        }

        appTimeout = globalConf.getInt(GlobalConfigurationKeys.APPLICATION_TIMEOUT,
                GlobalConfigurationKeys.DEFAULT_APPLICATION_TIMEOUT);

        localAppJarPath =  globalConf.get(GlobalConfigurationKeys.SHIFU_YARN_APP_JAR);
        LOG.info("localSelfJarPath:" + localAppJarPath);
        
        createYarnClient();
        return true;
    }

    private void createYarnClient() {
        int numRMConnectRetries = globalConf.getInt(GlobalConfigurationKeys.RM_CLIENT_CONNECT_RETRY_MULTIPLIER,
                GlobalConfigurationKeys.DEFAULT_RM_CLIENT_CONNECT_RETRY_MULTIPLIER);
        long rmMaxWaitMS = globalConf.getLong(YarnConfiguration.RESOURCEMANAGER_CONNECT_RETRY_INTERVAL_MS,
                YarnConfiguration.DEFAULT_RESOURCEMANAGER_CONNECT_RETRY_INTERVAL_MS) * numRMConnectRetries;
        globalConf.setLong(YarnConfiguration.RESOURCEMANAGER_CONNECT_MAX_WAIT_MS, rmMaxWaitMS);

        yarnClient = YarnClient.createYarnClient();
        yarnClient.init(yarnConf);
    }

    public void close() throws Exception {

    }

    public static void main(String[] args) {
        int exitCode = 0;
        TensorflowClient client = new TensorflowClient(new Configuration());
        boolean sanityCheck;

        try {
            sanityCheck = client.init(args);

            if(!sanityCheck) {
                LOG.fatal("Failed to init client.");
                exitCode = -1;
            } else {
                exitCode = client.start();
            }
        } catch (Exception e) {
            LOG.fatal("Encountered exception while initializing client or finishing application.", e);
            exitCode = -1;
        } finally {
            LOG.info("Closing Tensorflow client....");
            try {
                client.close();
            } catch (Exception e) {
                LOG.fatal("Client Close with error.", e);
            }
        }

        System.exit(exitCode);
    }

    /**
     * @return
     */
    private int start() {
        boolean result;
        try {
            result = run();
        } catch (Exception e) {
            LOG.fatal("Failed to run TensorflowClient", e);
            result = false;
        }
        if(result) {
            LOG.info("Application completed successfully");
            return 0;
        }
        LOG.error("Application failed to complete successfully");
        return -1;
    }

    /**
     * @return
     * @throws IOException
     * @throws YarnException
     * @throws InterruptedException
     */
    public boolean run() throws YarnException, IOException, InterruptedException {
        LOG.info("Starting client..");
        LOG.info("Hadoop env: " + System.getenv(Constants.HADOOP_CONF_DIR));
        // 1. Connect with Resource Manager
        yarnClient.start();

        YarnClientApplication app = yarnClient.createApplication();
        GetNewApplicationResponse appResponse = app.getNewApplicationResponse();

        int maxMem = appResponse.getMaximumResourceCapability().getMemory();

        // Truncate resource request to cluster's max resource capability.
        if(amMemory > maxMem) {
            LOG.warn("Truncating requested AM memory: " + amMemory + " to cluster's max: " + maxMem);
            amMemory = maxMem;
        }
        int maxVCores = appResponse.getMaximumResourceCapability().getVirtualCores();

        if(amVCores > maxVCores) {
            LOG.warn("Truncating requested AM vcores: " + amVCores + " to cluster's max: " + maxVCores);
            amVCores = maxVCores;
        }

        // Upload local resource like python env and glibc to hdfs
        FileSystem hdfs = HDFSUtils.getFS();
        ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
        ApplicationId appId = appContext.getApplicationId();
        
        appResourcesPath = Constants.getAppResourcePath(appId.toString());
        globalResourcesPath = Constants.getGlobalResourcePath();
        
        if(localPythonVenv != null) {
            uploadFileAndSetConfContainerResources(globalResourcesPath, new Path(localPythonVenv), Constants.PYTHON_VENV_ZIP, globalConf,
                    hdfs);
        }

        if(localGlibcVenv != null) {
            uploadFileAndSetConfContainerResources(globalResourcesPath, new Path(localGlibcVenv), Constants.GLIBC_VENV_ZIP, globalConf,
                    hdfs);
        }

        localGlobalFinalConfPath = Constants.getClientResourcesPath(appId.toString(), Constants.GLOBAL_FINAL_XML);

        OutputStream os = null;
        try {
            // Write user's overridden conf to an xml to be localized.
            os = new FileOutputStream(localGlobalFinalConfPath);
            globalConf.writeXml(os);
        } catch (IOException e) {
            throw new RuntimeException("Failed to create " + this.localGlobalFinalConfPath + " conf file. Exiting.", e);
        } finally {
            if(os != null) {
                os.close();
            }
        }

        // setting yarn resouce manager
        String appName = globalConf.get(GlobalConfigurationKeys.APPLICATION_NAME,
                GlobalConfigurationKeys.DEFAULT_APPLICATION_NAME);
        appContext.setApplicationName(appName);

        // Set up resource type requirements
        Resource capability = Resource.newInstance(amMemory, amVCores);
        appContext.setResource(capability);

        // Set the queue to which this application is to be submitted in the RM
        String yarnQueue = globalConf.get(GlobalConfigurationKeys.YARN_QUEUE_NAME,
                GlobalConfigurationKeys.DEFAULT_YARN_QUEUE_NAME);
        appContext.setQueue(yarnQueue);

        // Generate launch context includes command line
        // Set the ContainerLaunchContext to describe the Container ith which the TensorflowApplicationMaster is
        // launched.
        ContainerLaunchContext amSpec = createAMContainerSpec(appId);
        appContext.setAMContainerSpec(amSpec);

        LOG.info("Submitting YARN application");
        yarnClient.submitApplication(appContext);
        ApplicationReport report = yarnClient.getApplicationReport(appId);
        logTrackingAndRMUrls(report);
        return monitorApplication(appId);
    }

    public ContainerLaunchContext createAMContainerSpec(ApplicationId appId) throws IOException {
        ContainerLaunchContext amContainer = Records.newRecord(ContainerLaunchContext.class);

        // Add glabol conf, AM needed resource and container needed resource
        FileSystem hdfs = HDFSUtils.getFS();
        Map<String, LocalResource> localResources = new HashMap<String, LocalResource>();

        // Add global final conf into resources so that AM could read it
        uploadToHdfsAddIntoResourseMap(hdfs, 
                localGlobalFinalConfPath, 
                LocalResourceType.FILE, 
                new Path(this.appResourcesPath, Constants.GLOBAL_FINAL_XML),
                Constants.GLOBAL_FINAL_XML,
                localResources
                );

        // add lib jar that AM needed from hdfs to resource map
        uploadToHdfsAddIntoResourseMap(hdfs, 
                localJarLibPath, 
                LocalResourceType.ARCHIVE, 
                new Path(this.globalResourcesPath, Constants.JAR_LIB_PATH),
                "lib", // archive file will be unzip under "lib" folder
                localResources
                );
        
        // add self jar that AM needed from hdfs to resource map
        uploadToHdfsAddIntoResourseMap(hdfs, 
                localAppJarPath, 
                LocalResourceType.FILE, 
                new Path(this.appResourcesPath, new File(localAppJarPath).getName()),
                new File(localAppJarPath).getName(),
                localResources
                );
        
        // Set logs to be readable by everyone. Set app to be modifiable only by app owner.
        Map<ApplicationAccessType, String> acls = new HashMap<ApplicationAccessType, String>(2);
        acls.put(ApplicationAccessType.VIEW_APP, "*");
        acls.put(ApplicationAccessType.MODIFY_APP, " ");
        amContainer.setApplicationACLs(acls);

        // generate env map for running AM. it could also be part of env for running taskexecutor
        Map<String, String> AMExecutingEnv = generateAMEnvironment(hdfs, localResources);
        
        String command = buildCommand(amMemory, AMExecutingEnv);

        LOG.info("Completed setting up Application Master command " + command);
        amContainer.setCommands(ImmutableList.of(command));
        amContainer.setEnvironment(AMExecutingEnv);
        amContainer.setLocalResources(localResources);
        setToken(amContainer);
        
        return amContainer;
    }

    /**
     * Set delegation tokens for AM container
     * 
     * @param amContainer
     *            AM container
     */
    private void setToken(ContainerLaunchContext amContainer) throws IOException {
        // Setup security tokens
        if(UserGroupInformation.isSecurityEnabled()) {
            Credentials credentials = new Credentials();
            String tokenRenewer = yarnConf.get(YarnConfiguration.RM_PRINCIPAL);
            if(tokenRenewer == null || tokenRenewer.length() == 0) {
                throw new IOException("Can't get Master Kerberos principal for the RM to use as renewer");
            }
            
            // For now, only getting tokens for the default file-system.
            final Token<?>[] tokens = HDFSUtils.getFS().addDelegationTokens(tokenRenewer, credentials);
            if(tokens != null) {
                for(Token<?> token: tokens) {
                    LOG.info("Got dt for " + HDFSUtils.getFS().getUri() + "; " + token);
                }
            }
            DataOutputBuffer dob = new DataOutputBuffer();
            credentials.writeTokenStorageToStream(dob);
            ByteBuffer fsTokens = ByteBuffer.wrap(dob.getData(), 0, dob.getLength());
            amContainer.setTokens(fsTokens);
        }
    }
    
    static String buildCommand(long amMemory, Map<String, String> containerEnv) {
        List<String> arguments = new ArrayList<String>(30);
        arguments.add(HdfsUtils.$$(ApplicationConstants.Environment.JAVA_HOME.toString()) + "/bin/java");
        // Set Xmx based on am memory size
        arguments.add("-Xmx" + (int) (amMemory * 0.8f) + "m");
        // Add configuration for log dir to retrieve log output from python subprocess in AM
        arguments.add(
                "-D" + YarnConfiguration.YARN_APP_CONTAINER_LOG_DIR + "=" + ApplicationConstants.LOG_DIR_EXPANSION_VAR);

        // Set class name
        arguments.add(" " + TensorflowApplicationMaster.class.getName() + " ");
        
        for(Map.Entry<String, String> entry: containerEnv.entrySet()) {
            arguments.add("--container_env " + entry.getKey() + "=" + entry.getValue());
        }

        arguments.add(
                "1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + File.separatorChar + Constants.AM_STDOUT_FILENAME);
        arguments.add(
                "2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + File.separatorChar + Constants.AM_STDERR_FILENAME);
        return String.join(" ", arguments);
    }

    private  Map<String, String> generateAMEnvironment(FileSystem fs, 
            Map<String, LocalResource> localResources) throws IOException {
        // Add AppMaster.jar location to classpath
        // At some point we should not be required to add
        // the hadoop specific classpaths to the env.
        // It should be provided out of the box.
        // For now setting all required classpaths including
        // the classpath to "." for the application jar
        Map<String, String> amEnv = new HashMap<String, String>();
        
        StringBuilder classPathEnv = new StringBuilder(
                HdfsUtils.$$(ApplicationConstants.Environment.CLASSPATH.toString()))
                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./*")
                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/zip4j-1.3.2.jar")
                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/guagua-mapreduce-0.7.8-hadoop2.jar")
                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/guagua-core-0.7.8.jar")
                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/shifu-0.12.1-SNAPSHOT.jar");
        
        for(String c: yarnConf.getStrings(YarnConfiguration.YARN_APPLICATION_CLASSPATH,
                HdfsUtils.DEFAULT_YARN_CROSS_PLATFORM_APPLICATION_CLASSPATH)) {
            classPathEnv.append(HdfsUtils.CLASS_PATH_SEPARATOR);
            classPathEnv.append(c.trim());
        }
        amEnv.put("CLASSPATH", classPathEnv.toString());
        LOG.info("after setting classpath: " + amEnv.get("CLASSPATH"));
        return amEnv;
    }

    /**
     * Add a local resource to HDFS and local resources map.
     * 
     * @param hdfs
     *            HDFS file system reference
     * @param resourceType
     *            the type of the src file
     * @param dstPath
     *            name of the resource after localization
     * @param localResources
     *            the local resources map
     * @throws IOException
     *             error when writing to HDFS
     */
    private void uploadToHdfsAddIntoResourseMap(FileSystem hdfs, 
            String srcPath, 
            LocalResourceType resourceType, 
            Path dst,
            String resourceKey,
            Map<String, LocalResource> localResources) throws IOException {
        
        
        hdfs.copyFromLocalFile(new Path(srcPath), dst);
        hdfs.setPermission(dst, new FsPermission((short) 0770));
        FileStatus scFileStatus = hdfs.getFileStatus(dst);
        LocalResource scRsrc = LocalResource.newInstance(ConverterUtils.getYarnUrlFromURI(dst.toUri()), resourceType,
                LocalResourceVisibility.APPLICATION, scFileStatus.getLen(), scFileStatus.getModificationTime());
        localResources.put(resourceKey, scRsrc);
    }

    private void uploadFileAndSetConfContainerResources(Path hdfsPath, Path filePath, String fileName, Configuration gobalConf,
            FileSystem hdfs) throws IOException {
        Path dst = new Path(hdfsPath, fileName);
        if (!hdfs.exists(dst)) {
            hdfs.copyFromLocalFile(filePath, dst);
            hdfs.setPermission(dst, new FsPermission((short) 0770));
        }
        appendConfResources(GlobalConfigurationKeys.getContainerResourcesKey(), dst.toString(), gobalConf);
    }

    private void appendConfResources(String key, String resource, Configuration gobalConf) {
        String currentResources = gobalConf.get(key, "");
        gobalConf.set(GlobalConfigurationKeys.getContainerResourcesKey(), currentResources + "," + resource);
    }

    /**
     * Monitor the submitted application for completion.
     * Kill application if time expires.
     * 
     * @param appId
     *            Application Id of application to be monitored
     * @return true if application completed successfully
     * @throws org.apache.hadoop.yarn.exceptions.YarnException
     * @throws java.io.IOException
     */
    private boolean monitorApplication(ApplicationId appId) throws YarnException, IOException, InterruptedException {

        while(true) {
            // Check app status every 1 second.
            Thread.sleep(1000);

            // Get application report for the appId we are interested in
            ApplicationReport report = yarnClient.getApplicationReport(appId);

            YarnApplicationState state = report.getYarnApplicationState();
            FinalApplicationStatus dsStatus = report.getFinalApplicationStatus();

            if(YarnApplicationState.FINISHED == state || YarnApplicationState.FAILED == state
                    || YarnApplicationState.KILLED == state) {
                LOG.info("Application " + appId.getId() + " finished with YarnState=" + state.toString()
                        + ", DSFinalStatus=" + dsStatus.toString() + ", breaking monitoring loop.");
                String histHost = globalConf.get(GlobalConfigurationKeys.SHIFU_HISTORY_HOST,
                        GlobalConfigurationKeys.DEFAULT_SHIFU_HISTORY_HOST);
                CommonUtils.printTHSUrl(histHost, appId.toString(), LOG);
                return FinalApplicationStatus.SUCCEEDED == dsStatus;
            }

            if(appTimeout > 0) {
                if(System.currentTimeMillis() > (clientStartTime + appTimeout)) {
                    LOG.info("Reached client specified timeout for application. Killing application"
                            + ". Breaking monitoring loop : ApplicationId:" + appId.getId());
                    forceKillApplication(appId);
                    return false;
                }
            }
        }
    }

    private void logTrackingAndRMUrls(ApplicationReport report) {
        LOG.info("URL to track running application (will proxy to TensorBoard once it has started): "
                + report.getTrackingUrl());
        
          LOG.info("ResourceManager web address for application: "
          + CommonUtils.buildRMUrl(yarnConf, report.getApplicationId().toString()));
         
    }

    /**
     * Kill a submitted application by sending a call to the ASM
     * 
     * @param appId
     *            Application Id to be killed.
     * @throws org.apache.hadoop.yarn.exceptions.YarnException
     * @throws java.io.IOException
     */
    private void forceKillApplication(ApplicationId appId) throws YarnException, IOException {
        yarnClient.killApplication(appId);

    }
}
