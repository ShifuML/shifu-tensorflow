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

import java.nio.ByteBuffer;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;

/**
 * @author webai
 *
 */
public class NMCallbackHandler implements NMClientAsync.CallbackHandler {
    private static final Log LOG = LogFactory.getLog(NMCallbackHandler.class);

    public void onContainerStarted(ContainerId containerId, Map<String, ByteBuffer> allServiceResponse) {
        LOG.info("Successfully started container " + containerId);
    }

    public void onContainerStatusReceived(ContainerId containerId, ContainerStatus containerStatus) {
        LOG.info("Container Status: id =" + containerId + ", status =" + containerStatus);
    }

    public void onContainerStopped(ContainerId containerId) {
        LOG.info("Succeeded to stop container " + containerId);
    }

    public void onGetContainerStatusError(ContainerId containerId, Throwable throwable) {
        LOG.error("Failed to query the status of container " + containerId, throwable);
    }

    public void onStartContainerError(ContainerId containerId, Throwable t) {
        LOG.error("Failed to start container " + containerId, t);
    }

    public void onStopContainerError(ContainerId containerId, Throwable t) {
        LOG.error("Failed to stop container " + containerId, t);
    }

}
