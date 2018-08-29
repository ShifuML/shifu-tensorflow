package ml.shifu.tensorflow;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.yarn.client.api.async.*;
import org.apache.hadoop.yarn.api.records.*;
import org.apache.hadoop.yarn.api.*;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.util.ConverterUtils;

public class ShifuRMCallBackHandler implements AMRMClientAsync.CallbackHandler {

    ShifuTensorflowAM shifuAM;
    AtomicInteger counter = new AtomicInteger();
    
    public ShifuRMCallBackHandler(ShifuTensorflowAM shifuAM) {
        this.shifuAM = shifuAM;
    }

    public void onContainersAllocated(List<Container> containers) {
        for(Container container : containers) {
            Thread t = new Thread(new ContainerLauncher(container, shifuAM));
            t.start();
            counter.incrementAndGet();
            System.out.println("luanch container " + container.getId().getId() + " finished");
        }
    }

    public void onContainersCompleted(List<ContainerStatus> statuses) {
            if(counter.decrementAndGet() == 0) {
                shifuAM.setFinished(true);
            }
    }

    public void onNodesUpdated(List<NodeReport> updated) {

    }

    public void onReboot() {

    }

    public void onShutdownRequest() {
        shifuAM.setFinished(true);
        shifuAM.getAmClient().stop();
        shifuAM.getNmClient().stop();
    }

    public float getProgress() {
        return 0;
    }

    public void onError(Throwable e) {
        shifuAM.setFinished(true);
        shifuAM.getAmClient().stop();
        shifuAM.getNmClient().stop();
    }
    
    public static class ContainerLauncher implements Runnable {
        
        Container container;
        ShifuTensorflowAM am;
        
        public ContainerLauncher(Container container, ShifuTensorflowAM am) {
            this.container = container;
            this.am = am;
        }
        
        public void run() {
            try {
                List<String> commands = new ArrayList<String>();
                String command = "export;java -cp ./*: ml.shifu.tensorflow.ShifuWorkerTask";
                commands.add(command);
                commands.add("1> " + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/" + ApplicationConstants.STDOUT);
                commands.add("2> " + ApplicationConstants.LOG_DIR_EXPANSION_VAR + "/" + ApplicationConstants.STDERR);

                System.out.println("pos 9");
                //String command = "whoami";
                commands.add(command);

                Configuration conf = new Configuration();
                System.out.println("pos 1");
                FileSystem fs = FileSystem.get(conf);
                Path dist = fs.makeQualified(new Path("/tmp/" + am.getAppIdStr() + "/shifu-tensorflow-train-1.0-SNAPSHOT.jar"));
                System.out.println("pos 2");
                URL url = ConverterUtils.getYarnUrlFromPath(dist);
                System.out.println("pos 3");
                Map<String, LocalResource> localResource = new HashMap<String, LocalResource>();
                LocalResource res = LocalResource.newInstance(url, LocalResourceType.FILE, LocalResourceVisibility.APPLICATION, fs.getFileStatus(dist).getLen(), fs.getFileStatus(dist).getModificationTime());
                System.out.println("pos 4");
                localResource.put(dist.getName(), res);

                ContainerLaunchContext ctx = ContainerLaunchContext.newInstance(localResource, System.getenv(), commands, null,null, null);
                //ContainerLaunchContext ctx = ContainerLaunchContext.newInstance(localResource, System.getenv(), commands, null,null, null);
                System.out.println("pos 5");

                am.getNmClient().startContainerAsync(container, ctx);
                System.out.println("start worker task");
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
    }
}
