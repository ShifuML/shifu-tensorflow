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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.StringUtils;
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
import net.lingala.zip4j.core.ZipFile;
import net.lingala.zip4j.exception.ZipException;
import net.lingala.zip4j.model.ZipParameters;

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

    private String hdfsPythonVenv;
    private String hdfsGlibcVenv;
    
    // TODO: currently, we only use specific training script. this will be configurable in future
    // private String executes;
    //private String localPythonVenv = "/x/home/webai/tensorflow/python_env.zip";
    //private String localGlibcVenv = "/x/home/webai/tensorflow/glibc.zip";
    //private String localJarLibPath = "/x/home/webai/tensorflow/lib.zip";
    //private String localAppJarPath = null;

    private String localGlobalFinalConfPath = null;
    
    // resource folder on hdfs
    private Path appResourcesPath;
    
    private final long clientStartTime = System.currentTimeMillis();

    // Configurations
    private Configuration globalConf;
    private YarnConfiguration yarnConf; 
    
    public TensorflowClient(Configuration conf) {
        globalConf = conf;
        yarnConf = new YarnConfiguration();
        LOG.info("yarnConf:" + yarnConf.toString());
    }

    public TensorflowClient() {
        this(new Configuration(false));
    }
    
    public static CommandLine initOpts(String[] args) throws ParseException {
        Options opts = new Options();

        opts.addOption("libjars", true, "");
        opts.addOption("globalconfig", true, "");
        
        return new GnuParser().parse(opts, args);
    }
    
    /**
     * Collect all dependent jars from global conf and args
     */
    private List<String> allLibJars(CommandLine line) throws IOException {
        ArrayList<String> jars = new ArrayList<String>();
        
        String jarsFromConf = globalConf.get(GlobalConfigurationKeys.SHIFU_YARN_LIB_JAR, "");
        jars.addAll(validateFiles(jarsFromConf));
        
        String jarsFromArgs = line.getOptionValue("libjars");
        jars.addAll(validateFiles(jarsFromArgs));

        return jars;
    }
    
    /**
     * make these jar avalible to use in current java application
     */
    private void setJarsInCurrentClasspath(List<String> jars) throws IOException {
        List<URL> cp = new ArrayList<URL>();
        for (String jar : jars) {
            Path tmp = new Path(jar);
            cp.add(FileSystem.getLocal(globalConf).pathToFile(tmp).toURI().toURL());
        }
        
        if (!cp.isEmpty()) {
            globalConf.setClassLoader(new URLClassLoader(cp.toArray(new URL[0]), globalConf.getClassLoader()));
            Thread.currentThread().setContextClassLoader(
                    new URLClassLoader(cp.toArray(new URL[0]), Thread.currentThread().getContextClassLoader())); 
        }
    }
    
    public static void zipFiles(List<String> jars, String dst) throws IOException, ZipException {
        FileSystem fs = HDFSUtils.getLocalFS();
        Path dstPath = new Path(dst);
        if (fs.exists(dstPath)) {
            fs.delete(dstPath);
        }
        
        //fs.mkdirs(new Path(Constants.JAR_LIB_ROOT));

        ZipFile zipFile = new ZipFile(dst);
        ZipParameters zipParameters = new ZipParameters();
        //zipFile.addFolder(new File(Constants.JAR_LIB_ROOT), zipParameters);
        
        //zipParameters.setRootFolderInZip(Constants.JAR_LIB_ROOT);
        for (String jar : jars) {
            LOG.info(jar);
            LOG.info(jar.indexOf(':'));
            String path = jar.substring(jar.indexOf(':')+1);
            File jarFile = new File(path);
            //zipParameters.setDefaultFolderPath(path.substring(0, path.lastIndexOf('/')));
            zipFile.addFile(jarFile, zipParameters);
        }
        //fs.delete(new Path(Constants.JAR_LIB_ROOT));
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
        
        CommandLine line = initOpts(args);
        
        // get global conf path from args
        String globalConfPath;
        if (line.hasOption("globalconfig")) {
            globalConfPath = line.getOptionValue("globalconfig");
        } else {
            LOG.info("We use default global in tensorflow yarn jar instead of user's own.");
            globalConfPath = Constants.GLOBAL_DEFAULT_XML;
        }
        globalConf.addResource(globalConfPath);
        
        // collect all dependent jars 
        List<String> libjars = allLibJars(line);
        
        // setting jars in client classpath
        setJarsInCurrentClasspath(libjars);
        
        // cp jars into project lib folder and zip to a zip file
        zipFiles(libjars, Constants.JAR_LIB_ZIP);

        hdfsPythonVenv = globalConf.get(GlobalConfigurationKeys.PYTHON_ENV_ZIP);
        hdfsGlibcVenv = globalConf.get(GlobalConfigurationKeys.GLIBC_ENV_ZIP);
        
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
        TensorflowClient client = new TensorflowClient(new Configuration());
        boolean sanityCheck;

        try {
            sanityCheck = client.init(args);

            if(!sanityCheck) {
                LOG.fatal("Failed to init client.");
                throw new RuntimeException("Failed to init client.");
            } else {
                if (client.start() != 0) {
                    throw new RuntimeException("Executing tensorflow client fails");
                }
            }
        } catch (Exception e) {
            LOG.fatal("Encountered exception while initializing client or finishing application.", e);
            throw new RuntimeException("Failed to init client.", e);
        } finally {
            LOG.info("Closing Tensorflow client....");
            try {
                client.close();
            } catch (Exception e) {
                LOG.fatal("Client Close with error.", e);
                throw new RuntimeException("Failed to init client.", e);
            }
        }
    }

    /**
     * @return
     */
    public int start() {
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

        // Here we upload python script and shell to hdfs and prepare for using in each worker
        // python env, glibc are already in certain folder in hdfs already
        FileSystem hdfs = HDFSUtils.getFS();
        ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
        ApplicationId appId = appContext.getApplicationId();
        appResourcesPath = Constants.getAppResourcePath(appId.toString());
        
        // 
        if(hdfsPythonVenv != null) {
            setContainerResources(new Path(hdfsPythonVenv), globalConf);
        }

        if(hdfsGlibcVenv != null) {
            setContainerResources(new Path(hdfsGlibcVenv), globalConf);
        }

        uploadFileAndSetConfContainerResources(new Path(globalConf.get(GlobalConfigurationKeys.PYTHON_SCRIPT_PATH)),
                appResourcesPath, 
                globalConf,
                hdfs);
        
        uploadFileAndSetConfContainerResources(new Path(globalConf.get(GlobalConfigurationKeys.PYTHON_SHELL_PATH)),
                appResourcesPath, 
                globalConf,
                hdfs);
        
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
                Constants.JAR_LIB_ZIP, 
                LocalResourceType.ARCHIVE, 
                new Path(this.appResourcesPath, Constants.JAR_LIB_ZIP),
                Constants.JAR_LIB_ROOT, // archive file will be unzip under "lib" folder
                localResources
                );
        
        HDFSUtils.getLocalFS().delete(new Path(Constants.JAR_LIB_ZIP));
        
//          uploadToHdfsAddIntoResourseMap(hdfs, 
//              "/hadoop/home/webai/tensorflow-shifu/shifu-0.12.1-SNAPSHOT/lib/shifu-tensorflow-on-yarn-0.0.1-SNAPSHOT.jar", 
//              LocalResourceType.FILE, 
//              new Path(this.appResourcesPath, "shifu-tensorflow-on-yarn-0.0.1-SNAPSHOT.jar"),
//              "shifu-tensorflow-on-yarn-0.0.1-SNAPSHOT.jar", // archive file will be unzip under "lib" folder
//              localResources
//          );
        

        
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
        arguments.add("exec");
        
        arguments.add(HdfsUtils.$$(ApplicationConstants.Environment.JAVA_HOME.toString()) + "/bin/java");
        // Set Xmx based on am memory size
        arguments.add("-Xmx" + (int) (amMemory * 0.8f) + "m");
        
        arguments.add("-cp .:${CLASSPATH}");
        
        // Add configuration for log dir to retrieve log output from python subprocess in AM
        arguments.add(
                "-D" + YarnConfiguration.YARN_APP_CONTAINER_LOG_DIR + "=" + ApplicationConstants.LOG_DIR_EXPANSION_VAR);

        // Set class name
        arguments.add(" " + TensorflowApplicationMaster.class.getName() + " ");
        //arguments.add(" " + HelloWorld.class.getName() + " ");
        
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
                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./" + Constants.JAR_LIB_ROOT + "/*");
//                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/shifu-tensorflow-on-yarn-0.0.1-SNAPSHOT.jar");
//                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/zip4j-1.3.2.jar")
//                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/guagua-mapreduce-0.7.8-hadoop2.jar")
//                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/guagua-core-0.7.8.jar")
//                        .append(HdfsUtils.CLASS_PATH_SEPARATOR).append("./lib/shifu-0.12.1-SNAPSHOT.jar");
        
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

        LocalResource scRsrc = LocalResource.newInstance(ConverterUtils.getYarnUrlFromURI(hdfs.makeQualified(dst).toUri()), resourceType,
                LocalResourceVisibility.APPLICATION, scFileStatus.getLen(), scFileStatus.getModificationTime());
        
        localResources.put(resourceKey, scRsrc);
    }

    /**
     * 
     * @param src file
     * @param hdfsdst folder
     * @param gobalConf
     * @param hdfs
     * @throws IOException
     */
    private void uploadFileAndSetConfContainerResources(Path src, Path hdfsdst, Configuration gobalConf,
            FileSystem hdfs) throws IOException {
        Path dst = new Path(hdfsdst, src.getName());
        if (!hdfs.exists(dst)) {
            hdfs.copyFromLocalFile(src, dst);
            hdfs.setPermission(dst, new FsPermission((short) 0770));
        }
        appendConfResources(GlobalConfigurationKeys.getContainerResourcesKey(), dst.toString(), gobalConf);
    }
    
    private void setContainerResources(Path hdfsPath, Configuration gobalConf) {
        appendConfResources(GlobalConfigurationKeys.getContainerResourcesKey(), hdfsPath.toString(), gobalConf);
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
    
    private static final String FILE_SEPERATOR = ",";
    /**
     * Take input as a comma separated list of files and verifies if they exist. It defaults for file:/// if the files
     * specified do not have a scheme. it returns the paths uri converted defaulting to file:///. So an input of
     * /home/user/file1,/home/user/file2 would return file:///home/user/file1,file:///home/user/file2
     */
    private List<String> validateFiles(String files) throws IOException {
        ArrayList<String> finalArr = new ArrayList<String>();
        
        if(StringUtils.isBlank(files))
            return finalArr;
        
        String[] fileArr = files.split(FILE_SEPERATOR);
        
        for(int i = 0; i < fileArr.length; i++) {
            String tmp = fileArr[i];
            URI pathURI;
            try {
                pathURI = new URI(tmp);
            } catch (URISyntaxException e) {
                throw new IllegalArgumentException(e);
            }
            Path path = new Path(pathURI.toString());
            FileSystem localFs = HDFSUtils.getLocalFS();
            if(pathURI.getScheme() == null) {
                // default to the local file system
                // check if the file exists or not first
                if(!localFs.exists(path)) {
                    throw new FileNotFoundException("File " + tmp + " does not exist.");
                }
                finalArr.add(path.makeQualified(localFs).toString());
            }
        }
        return finalArr;
    }
}
