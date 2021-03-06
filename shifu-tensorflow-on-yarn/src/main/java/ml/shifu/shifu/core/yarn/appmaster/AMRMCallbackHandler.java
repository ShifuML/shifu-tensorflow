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
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.records.ApplicationAccessType;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerExitStatus;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.security.AMRMTokenIdentifier;
import org.apache.hadoop.yarn.util.AbstractLivelinessMonitor;

import com.google.common.base.Preconditions;

import ml.shifu.shifu.core.yarn.appmaster.TensorflowSession.SessionState;
import ml.shifu.shifu.core.yarn.util.CommonUtils;
import ml.shifu.shifu.core.yarn.util.Constants;
import ml.shifu.shifu.core.yarn.util.GlobalConfigurationKeys;
import ml.shifu.shifu.util.HDFSUtils;

/**
 * {@link AMRMCallbackHandler} 
 * 
 * @author webai
 *
 */
public class AMRMCallbackHandler implements AMRMClientAsync.CallbackHandler {
    private static final Log LOG = LogFactory.getLog(AMRMCallbackHandler.class);

    // cannot be used in construct function
    private TensorflowSession session;

    private Map<String, LocalResource> containerResources;

    private static FileSystem hdfs = HDFSUtils.getFS();

    private NMClientAsync nmClientAsync = null;

    private final AbstractLivelinessMonitor<TensorflowTask> hbMonitor;

    private Map<String, String> containerEnv;

    private ByteBuffer allTokens;

    private int lastRunEpochs = -1;

    // resource folder on hdfs
    private Path appResourcesPath;

    public AMRMCallbackHandler(Configuration globalConf, TensorflowSession session, NMClientAsync nmClientAsync,
            AbstractLivelinessMonitor<TensorflowTask> hbMonitor, Map<String, String> containerEnv, String appId) {
        this.session = session;
        this.containerResources = new ConcurrentHashMap<String, LocalResource>();
        this.nmClientAsync = nmClientAsync;
        this.hbMonitor = hbMonitor;
        this.containerEnv = containerEnv;

        appResourcesPath = Constants.getAppResourcePath(appId);

        // All resources available to all containers
        String[] resources = globalConf.getStrings(GlobalConfigurationKeys.getContainerResourcesKey());
        if(null != resources) {
            for(String dir: resources) {
                CommonUtils.addResource(dir, containerResources, hdfs);
            }
        }
        // Add global conf to resource
        CommonUtils.addResource(new Path(this.appResourcesPath, Constants.GLOBAL_FINAL_XML), containerResources, hdfs,
                LocalResourceType.FILE, Constants.GLOBAL_FINAL_XML);

        // Add lib jar from hdfs to local resource
        CommonUtils.addResource(new Path(this.appResourcesPath, Constants.JAR_LIB_ZIP), containerResources, hdfs,
                LocalResourceType.ARCHIVE, Constants.JAR_LIB_ROOT);

        getAllTokens();
    }

    public void onContainersCompleted(List<ContainerStatus> completedContainers) {
        LOG.info("Completed containers: " + completedContainers.size());
        for(ContainerStatus containerStatus: completedContainers) {
            int exitStatus = containerStatus.getExitStatus();
            LOG.info("ContainerID = " + containerStatus.getContainerId() + ", state = " + containerStatus.getState()
                    + ", exitStatus = " + exitStatus);
            String diagnostics = containerStatus.getDiagnostics();
            if(ContainerExitStatus.SUCCESS != exitStatus) {
                LOG.error(diagnostics);
            } else {
                LOG.info(diagnostics);
            }
            TensorflowTask task = session.getTaskByContainerId(containerStatus.getContainerId());
            if(task != null) {
                LOG.warn("container : [" + containerStatus.getContainerId() + "] isregister!" + task.isRegister());
                if(!task.isRegister()) {
                    LOG.warn("container : [" + containerStatus.getContainerId() + "] does not register!");
                    continue;
                }
                // Update Tensorflow Session on the state of the task.
                session.onTaskCompleted(task.getJobName(), task.getTaskIndex(), exitStatus);

                // Unregister task after completion..
                // Since in the case of asynchronous exec, containers might
                // end at different times..
                LOG.info("Unregister task [" + task.getId() + "] from Heartbeat monitor..");
                hbMonitor.unregister(task);
            } else {
                LOG.warn("No task found for container : [" + containerStatus.getContainerId() + "]!");
            }
        }
    }

    public void onContainersAllocated(List<Container> containers) {
        LOG.info("Allocated: " + containers.size() + " containers.");
        for(Container container: containers) {
            LOG.info("Launching a task in container" + ", 3 = " + container.getId() + ", containerNode = "
                    + container.getNodeId().getHost() + ":" + container.getNodeId().getPort() + ", resourceRequest = "
                    + container.getResource());
            TensorflowTask task = session.distributeTaskToContainer(container);
            Preconditions.checkNotNull(task, "Task was null! Nothing to schedule.");

            CommonUtils.printTaskUrl(task.getTaskUrl(), LOG);

            List<String> commands = new ArrayList<String>();
            List<CharSequence> arguments = new ArrayList<CharSequence>(5);

            arguments.add(task.getTaskCommand());
            arguments.add("1>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stdout");
            arguments.add("2>" + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/stderr");
            StringBuilder command = new StringBuilder();
            for(CharSequence str: arguments) {
                command.append(str).append(" ");
            }
            commands.add(command.toString());
            LOG.info("Constructed command: " + commands);

            // Set logs to be readable by everyone.
            Map<ApplicationAccessType, String> acls = new HashMap<ApplicationAccessType, String>(2);
            acls.put(ApplicationAccessType.VIEW_APP, "*");
            acls.put(ApplicationAccessType.MODIFY_APP, " ");

            ContainerLaunchContext ctx = ContainerLaunchContext.newInstance(containerResources, containerEnv, commands,
                    null, null, acls);

            ctx.setTokens(this.allTokens.slice());

            nmClientAsync.startContainerAsync(container, ctx);
        }

        // if all task have container, it mean session will go to next stage which is register cluster
        if(session.isAllTaskAssignedContainer()) {
            session.setState(SessionState.REGESTERING_CLUSTER);
            session.setStartTimeOfRegisteringCluster(System.currentTimeMillis());
            LOG.info("Session goes to REGESTERING_CLUSTER");
        }
    }

    /*
     * Populate allTokens with the tokens received
     */
    private void getAllTokens() {
        Credentials credentials;
        try {
            credentials = UserGroupInformation.getCurrentUser().getCredentials();
            DataOutputBuffer dob = new DataOutputBuffer();
            credentials.writeTokenStorageToStream(dob);

            // Now remove the AM->RM token so that containers cannot access it.
            Iterator<Token<?>> iter = credentials.getAllTokens().iterator();
            while(iter.hasNext()) {
                Token<?> token = iter.next();
                LOG.info("Token type : " + token.getKind());
                if(token.getKind().equals(AMRMTokenIdentifier.KIND_NAME)) {
                    iter.remove();
                }
            }
            this.allTokens = ByteBuffer.wrap(dob.getData(), 0, dob.getLength());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void onShutdownRequest() {
    }
    
    public void onNodesUpdated(List<NodeReport> updatedNodes) {
    }

    public float getProgress() {
        if(lastRunEpochs == -1 && session.getGlobalEpoch().get() > 1) {
            // first global epoch is not 1, continuous training to get last global epoch
            this.lastRunEpochs = session.getGlobalEpoch().get();
        }

        if(lastRunEpochs == -1) {
            if(session.getGlobalEpoch().get() > session.getTotalEpochs()) {
                return 1f;
            }
            return (float) session.getGlobalEpoch().get() / session.getTotalEpochs();
        } else {
            if(session.getGlobalEpoch().get() > (this.lastRunEpochs - 1)) {
                float progress = (float) (session.getGlobalEpoch().get() - this.lastRunEpochs + 1)
                        / session.getTotalEpochs();
                return progress > 1f ? 1f : progress;
            } else {
                return 0f;
            }
        }
    }

    public void onError(Throwable e) {
        LOG.error("Error: stop nmClientAsync", e);
        nmClientAsync.stop();
    }
}
