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
package ml.shifu.shifu.core.yarn.container;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.ZooDefs.Ids;

import ml.shifu.guagua.coordinator.zk.GuaguaZooKeeper;
import ml.shifu.shifu.core.TrainingIntermediateResult;
import ml.shifu.shifu.core.yarn.util.Constants;

/**
 * This server is used to connect with python program to collect worker training intermediate result:
 *   training error
 *   valid error
 *   execution time of each epoch
 * 
 * @author webai
 */
public class SocketServer extends Thread{
    private static final Log LOG = LogFactory.getLog(SocketServer.class);
    
    private ServerSocket server;
    private GuaguaZooKeeper zookeeper;
    private String containerId;
    
    public SocketServer(GuaguaZooKeeper zookeeper, String containerId) throws IOException {
        server = new ServerSocket(0);
        this.zookeeper = zookeeper;
        this.containerId = containerId;
    }
    
    public void run(){
        while(true) {
            try {
                Socket client = server.accept();
                LOG.info("got connection on port " + getServerPort());
                
                BufferedReader in = new BufferedReader(new InputStreamReader(client.getInputStream()));
                String rawMessage = in.readLine();
                
                while(StringUtils.isNotBlank(rawMessage)) {
                    LOG.info("received: " + rawMessage);
                    
                    // parse raw message to give to app master by zookeeper
                    TrainingIntermediateResult intermediate = new TrainingIntermediateResult();
                    String[] fields = rawMessage.split(",");
                    for (String field: fields) {
                        String[] keyValue = field.split(":", 2);
                        if ("worker_index".equals(keyValue[0])) {
                          intermediate.setWorkerIndex(Integer.valueOf(keyValue[1]));
                        } else if ("time".equals(keyValue[0])) {
                          intermediate.setCurrentEpochTime(Double.valueOf(keyValue[1]));
                        } else if ("current_epoch".equals(keyValue[0])) {
                          intermediate.setCurrentEpochStep(Integer.valueOf(keyValue[1])); 
                        } else if ("training_loss".equals(keyValue[0])) {
                          intermediate.setTrainingError(Double.valueOf(keyValue[1]));
                        } else if ("valid_loss".equals(keyValue[0])) {
                          intermediate.setValidError(Double.valueOf(keyValue[1]));
                        } else {
                          LOG.warn("There is unexpacted field in message: " + field);  
                        }
                    }
                    LOG.info("After Parsing: " + intermediate.toString());
                    zookeeper.createOrSetExt(Constants.WORKER_INTERMEDIATE_RESULT_ROOT_PATH + containerId, 
                            intermediate.serialize(), Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT, true, -1);
                    
                    rawMessage = in.readLine();
                }
            } catch (IOException e) {
                LOG.error("Scoket Server has some problem", e);
            } catch (KeeperException e) {
                LOG.error("Writing zookeeper has some problem", e);
            } catch (InterruptedException e) {
                LOG.error("Writing zookeeper has some problem", e);
            }
        }
    }
    
    public int getServerPort() {
        return server.getLocalPort();
    }
}
