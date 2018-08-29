package ml.shifu.tensorflow;

import java.util.*;
import java.net.*;
import java.nio.ByteBuffer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.*;

public class ShifuTensorflowAM {

    public static void main(String[] args) throws Throwable {
        ShifuTensorflowAM am = new ShifuTensorflowAM();
        am.setAppIdStr(args[0]);
        am.init(System.getenv());
        am.start();

    }
    
    private AMRMClientAsync<ContainerRequest> amClient;
    private NMClientAsync nmClient;
    private AMRMClientAsync.CallbackHandler rmCallBackHandler;
    private NMClientAsync.CallbackHandler nmCallBackHandler;
    private String appIdStr;
    private ByteBuffer tokens;
    
    private boolean finished = false;
    
    public AMRMClientAsync<ContainerRequest> getAmClient() {
        return amClient;
    }

    public void setAmClient(AMRMClientAsync<ContainerRequest> amClient) {
        this.amClient = amClient;
    }

    public NMClientAsync getNmClient() {
        return nmClient;
    }

    public void setNmClient(NMClientAsync nmClient) {
        this.nmClient = nmClient;
    }

    public AMRMClientAsync.CallbackHandler getRmCallBackHandler() {
        return rmCallBackHandler;
    }

    public void setRmCallBackHandler(AMRMClientAsync.CallbackHandler rmCallBackHandler) {
        this.rmCallBackHandler = rmCallBackHandler;
    }

    public NMClientAsync.CallbackHandler getNmCallBackHandler() {
        return nmCallBackHandler;
    }

    public void setNmCallBackHandler(NMClientAsync.CallbackHandler nmCallBackHandler) {
        this.nmCallBackHandler = nmCallBackHandler;
    }

    public boolean isFinished() {
        return finished;
    }

    public void setFinished(boolean finished) {
        this.finished = finished;
    }

    public void setAppIdStr(String id) {
        this.appIdStr = id;
    }

    public String getAppIdStr() {
        return this.appIdStr;
    }

    public void init(Map<String, String> env) throws Throwable {
        
//        String containerIdString = env.get(ApplicationConstants.AM_CONTAINER_ID_ENV);
//        ContainerId containerId = ConverterUtils.toContainerId(containerIdString);

        UserGroupInformation amUserGourpInfo =
                UserGroupInformation.createRemoteUser(System.getenv(ApplicationConstants.Environment.USER.name()));

        String hostName = InetAddress.getLocalHost().getHostName();
        System.out.println("net address : " + hostName);
        
        Configuration conf = new Configuration(); 
        //init call back handler
        AMRMClientAsync.CallbackHandler rmCallBackHandler = new ShifuRMCallBackHandler(this);
        this.rmCallBackHandler = rmCallBackHandler;
        AMRMClientAsync<ContainerRequest> rmClient = AMRMClientAsync.createAMRMClientAsync(1000, rmCallBackHandler);
        this.amClient = rmClient;
        
        rmClient.init(conf);
        rmClient.start();
        System.out.println("amClient start finished");
        

        NMClientAsync.CallbackHandler nmCallBackHandler = new ShifuNMCallBackHandler();
        this.nmCallBackHandler = nmCallBackHandler;
        NMClientAsync nmClient = NMClientAsync.createNMClientAsync(nmCallBackHandler);
        this.nmClient = nmClient;
        
        nmClient.init(conf);
        nmClient.start();
        System.out.println("nmClient start finished");
        //register to RM heart beat 
        RegisterApplicationMasterResponse response = rmClient.registerApplicationMaster(
            hostName, 12345, "");      
        System.out.println("register heart beat finished");
    }

    public void start() {
        
        int containerNum = 2;
        for(int i = 0; i < containerNum; i++) {
            System.out.println("ask RM for container : " + i);
            ContainerRequest containerTask = setContainerResource();        
            this.amClient.addContainerRequest(containerTask);
        }
        while(!finished) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public ContainerRequest setContainerResource() {

        Priority pri = Priority.newInstance(Integer.MAX_VALUE);
        int containerMemory = 1024;
        int containerVirtualCores = 1;
        Resource capability = Resource.newInstance(containerMemory, containerVirtualCores);
        ContainerRequest request = new ContainerRequest(capability, null, null, pri);

        return request;
    }
    
    private void shutdown() {  
        this.amClient.stop();
        this.nmClient.stop();
    }

}
