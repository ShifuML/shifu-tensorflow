package ml.shifu.tensorflow;

import java.nio.ByteBuffer;
import java.util.Map;

import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync.CallbackHandler;

public class ShifuNMCallBackHandler implements CallbackHandler {

    public void onContainerStarted(ContainerId containerId, Map<String, ByteBuffer> allServiceResponse) {
        System.out.println("container start : " + containerId.getId());
    }

    public void onContainerStatusReceived(ContainerId containerId, ContainerStatus containerStatus) {
        System.out.println("container " + containerId.getId() + " status " + containerStatus.getState().name());
    }

    public void onContainerStopped(ContainerId containerId) {
        
    }

    public void onStartContainerError(ContainerId containerId, Throwable t) {

    }

    public void onGetContainerStatusError(ContainerId containerId, Throwable t) {

    }

    public void onStopContainerError(ContainerId containerId, Throwable t) {

    }

}
