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
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;

import ml.shifu.shifu.core.yarn.util.CommonUtils;


/**
 * {@link AbstractApplicationMaster} Abstract App master to define workflow
 * @author webai
 */
public abstract class AbstractApplicationMaster {
    private static final Log LOG = LogFactory.getLog(AbstractApplicationMaster.class);
    
    /** Node Manager **/
    protected NMClientAsync nmClientAsync;
    /** Resource manager **/
    protected AMRMClientAsync<ContainerRequest> amRMClient;
    
    /** For status update for clients, Hostname of the container **/
    private String appMasterHostname;
    /** Port on which the app master listens for status updates from clients */
    private int appMasterRpcPort = 1234;
    
    public AbstractApplicationMaster() {
        appMasterHostname = CommonUtils.getCurrentHostName();
    }
    
    /**
     * init need variale by main arguments and conf files
     */
    protected abstract void init(String[] args);
    
    /**
     * Register RM Callback Handler
     */
    protected abstract void registerRMCallbackHandler();
    
    /**
     * Register NM Callback Handler
     */
    protected abstract void registerNMCallbackHandler();
    
    protected void registerAMToRM() {
        try {
            amRMClient.registerApplicationMaster(appMasterHostname, appMasterRpcPort, null);
        } catch (Exception e) {
            throw new IllegalStateException("AppMaster failed to register with RM.", e);
        }
    }
    
    /**
     * preparation work before schduleing task exector
     */
    protected abstract void prepareBeforeTaskExector();
    
    protected abstract void scheduleTask();
    
    /**
     * recursively monitor executor
     * @return
     *      false if some executor failed, return true if all executor complete
     */
    protected abstract boolean monitor();
    
    /**
     * After out of monitor, we need to summary overall training job success or not
     */
    protected abstract void updateTaskStatus();
    
    protected abstract boolean canRecovered();
    
    protected abstract void recovery();
    
    protected abstract void stop();
    
    /**
     * Main entrance of start app master
     * @param args
     */
    public void run(String[] args) {
        LOG.info("Start init....");
        init(args);
        
        LOG.info("Start registerNMCallbackHandler....");
        registerNMCallbackHandler();
        
        LOG.info("Start registerRMCallbackHandler....");
        registerRMCallbackHandler();
        
        LOG.info("Start registerAMToRM....");
        registerAMToRM();
        
        LOG.info("Start prepareBeforeTaskExector....");
        prepareBeforeTaskExector();
        
        LOG.info("Start scheduleTask....");
        scheduleTask();
        
        LOG.info("Start monitor....");
        while(!monitor()) {
            if (canRecovered()) {
                LOG.info("Start recovery....");
                recovery();
            } else {
                LOG.info("Cannot recover....");
                break;
            }
        }
        
        updateTaskStatus();
        
        LOG.info("Start stop....");
        stop();
    }
}
