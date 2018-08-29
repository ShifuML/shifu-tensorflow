package ml.shifu.tensorflow;

import org.apache.hadoop.yarn.client.api.*;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.util.ConverterUtils;
import java.util.List;
import java.util.Map;
import java.io.File;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.conf.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DataOutputBuffer;
import org.apache.hadoop.security.Credentials;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.yarn.api.*;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.protocolrecords.*;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationReport;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.URL;
import org.apache.hadoop.yarn.api.records.YarnApplicationState;

public class TrainClient {
    
    private YarnClient yc;
    private Configuration conf;
    private int reportCounter;
    private ApplicationId appId;
    private ByteBuffer tokens;

    public void run() throws Throwable {
        YarnClient yc = YarnClient.createYarnClient();
        this.yc = yc;
        Configuration conf = new Configuration();
        this.conf = conf;
        yc.init(conf);
        yc.start();

        YarnClientApplication app = yc.createApplication();
        String trainScriptPath = "./train.py";

        GetNewApplicationResponse appResponse = app.getNewApplicationResponse();
        this.appId = appResponse.getApplicationId();
        Resource res = appResponse.getMaximumResourceCapability();

        System.out.println("memory " + res.getMemory());
        System.out.println("vcores " + res.getVirtualCores());

        ApplicationSubmissionContext appContext = app.getApplicationSubmissionContext();
        appContext.setApplicationName("shifu_tensorflow_training");

        initTokens();

        appContext.setMaxAppAttempts(2);
        appContext.setAMContainerSpec(buildLauchContext());
        
        Resource reqRes = Resource.newInstance(1024, 4);
        
        appContext.setResource(reqRes);
                
        System.out.println("Submitting app");
        yc.submitApplication(appContext);
        
        boolean done = false;
        ApplicationReport report = null;
        try {
            do {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException ir) {
                    Thread.currentThread().interrupt();
                }
                report = this.yc.getApplicationReport(this.appId);
                YarnApplicationState jobState = report.getYarnApplicationState();
                int count = 0;
                switch(jobState) {
                    case FINISHED:
                    case KILLED:
                    case FAILED:
                        done = true;
                        break;
                    default:
                }
                count++;
                if(count == 10) {
                    System.out.println(jobState.toString());
                    count = 0;
                }
            } while(!done);
        } catch (Exception e) {
            e.printStackTrace();
        }
        removeHdfsFile();
        System.exit(0);
    }
    
    public ContainerLaunchContext buildLauchContext() throws Exception {
        Map<String, LocalResource> localResources = getLocalResource();
        ContainerLaunchContext ctx = ContainerLaunchContext.newInstance(localResources, System.getenv(), buildCommand(), null, null, null); 
        ctx.setTokens(tokens);
        return ctx;
    }
    
    public Map<String, LocalResource> getLocalResource() throws Exception {
        Map<String, LocalResource> localResource = new HashMap<String, LocalResource>();
        File curDir = new File("./");
        File[] files = curDir.listFiles();
        FileSystem fs = FileSystem.get(conf);
        FileSystem localFs = FileSystem.getLocal(conf);
        for(File f : files) {
            if(f.getName().endsWith("jar")) {
                Path src = localFs.makeQualified(new Path(f.getAbsolutePath()));
                Path dist = fs.makeQualified(new Path("/tmp/" + this.appId.toString() + "/" + f.getName()));
                fs.copyFromLocalFile(src, dist);
                URL url = ConverterUtils.getYarnUrlFromPath(dist);
                LocalResource res = LocalResource.newInstance(url, LocalResourceType.FILE, LocalResourceVisibility.APPLICATION, fs.getFileStatus(dist).getLen(), fs.getFileStatus(dist).getModificationTime());
                localResource.put(dist.getName(), res);
            }
        }
        return localResource;
    }
    
    private void removeHdfsFile() throws Exception {
        FileSystem fs = FileSystem.get(conf);
        Path path = fs.makeQualified(new Path("/tmp/" + this.appId.toString()));
        fs.delete(path, true);
    }

    public List<String> buildCommand() {
        List<String> commands = new ArrayList<String>();
        commands.add("exec");
        commands.add(Environment.JAVA_HOME.$() + "/bin/java");
        commands.add("-Xms1024m");
        commands.add("-Xmx1024m");
        commands.add("-cp ./*:${HADOOP_CONF_DIR}:");
        commands.add("ml.shifu.tensorflow.ShifuTensorflowAM");
        commands.add(this.appId.toString());
        commands.add("1> " + ApplicationConstants.LOG_DIR_EXPANSION_VAR + File.separator + ApplicationConstants.STDOUT);
        commands.add("2> " + ApplicationConstants.LOG_DIR_EXPANSION_VAR + File.separator + ApplicationConstants.STDERR);
        return commands;
    }

    public void initTokens() throws Throwable {
        if(UserGroupInformation.isSecurityEnabled()) {
            Credentials cred = UserGroupInformation.getCurrentUser().getCredentials();
            String tokenRenew = this.conf.get(YarnConfiguration.RM_PRINCIPAL);
            if(tokenRenew == null || tokenRenew.length() == 0) {
                System.out.println("Can not get kerberos  principal for RM to use as renew");
            } else {
                FileSystem fs = FileSystem.get(conf);
                final Token[] tokens = fs.addDelegationTokens(tokenRenew, cred);
                DataOutputBuffer buf = new DataOutputBuffer();
                cred.writeTokenStorageToStream(buf);
                this.tokens = ByteBuffer.wrap(buf.getData(), 0, buf.getLength());
            }
        }
    }
    
    public static void main(String[] args) throws Throwable {
        
        TrainClient tc = new TrainClient();
        tc.run();
    }
}
