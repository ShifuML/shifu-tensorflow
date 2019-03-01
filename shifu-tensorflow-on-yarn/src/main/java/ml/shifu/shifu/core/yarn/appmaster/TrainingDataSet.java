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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;

import ml.shifu.shifu.util.HDFSUtils;

/**
 * @author webai
 * So far, Splitting algorithm is very simple, we only try to split by file number, not considering file size
 */
public class TrainingDataSet {
    private int workerNum = 0;
    private String rootDataPath = null;
    private List<StringBuilder> splitedFilePaths = null;
    private Configuration globalConf;
    
    private FileSystem hdfs = HDFSUtils.getFS();
    
    private static class TrainingDataSetHolder {
        private static final TrainingDataSet INSTANCE = new TrainingDataSet();
    }
    
    private TrainingDataSet() {}
    
    public static final TrainingDataSet getInstance() {
        return TrainingDataSetHolder.INSTANCE;
    }
    
    public List<StringBuilder> getSplitedFilePaths(Configuration globalConf, int workerNum, String rootDataPath) throws FileNotFoundException, IllegalArgumentException, IOException {
        if (this.splitedFilePaths != null || this.globalConf != null) {
            return this.splitedFilePaths;
        }
        
        this.globalConf = globalConf;
        this.workerNum = workerNum;
        this.rootDataPath = rootDataPath;
        this.splitedFilePaths = new ArrayList<StringBuilder>();
        
        RemoteIterator<LocatedFileStatus> itr = hdfs.listFiles(new Path(rootDataPath), true);
        int cursor = 0;
        while(itr.hasNext()) {
            LocatedFileStatus data = itr.next();
            if (data.getPath().getName().startsWith(".") || data.getPath().getName().startsWith("_")) {
                continue;
            }
            
            if (this.splitedFilePaths.size() <= cursor+1) {
                this.splitedFilePaths.add(new StringBuilder(data.getPath().toString()));
            } else {
                this.splitedFilePaths.get(cursor)
                    .append(",")
                    .append(data.getPath().toString());
            }
            
            cursor = (cursor + 1) % this.workerNum;
        }
        
        if (splitedFilePaths.size() < workerNum) {
            throw new RuntimeException("Training data file count is smaller than worker number, this will make some workers do not have training data!");
        }
        
        return splitedFilePaths;
    }

}
