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
package ml.shifu.shifu.core.yarn.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;

import org.apache.commons.lang3.StringUtils;
import org.apache.hadoop.fs.ContentSummary;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.yarn.api.ApplicationConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author webai
 *
 */
public class HdfsUtils {
    private final static Logger LOG = LoggerFactory.getLogger(HdfsUtils.class);

    /**
     * The following two constants are used to expand parameter and it will be
     * replaced with real parameter expansion marker ('%' for Windows and '$' for
     * Linux) by NodeManager on container launch. For example: {{VAR}} will be
     * replaced as $VAR on Linux, and %VAR% on Windows. User has to use this
     * constant to construct class path if user wants cross-platform practice i.e.
     * submit an application from a Windows client to a Linux/Unix server or vice
     * versa.
     */
    public static final String PARAMETER_EXPANSION_LEFT="{{";
    
    /**
     * User has to use this constant to construct class path if user wants
     * cross-platform practice i.e. submit an application from a Windows client to
     * a Linux/Unix server or vice versa.
     */
    public static final String PARAMETER_EXPANSION_RIGHT="}}";
    
    /**
     * Expand the environment variable in platform-agnostic syntax. The
     * parameter expansion marker "{{VAR}}" will be replaced with real parameter
     * expansion marker ('%' for Windows and '$' for Linux) by NodeManager on
     * container launch. For example: {{VAR}} will be replaced as $VAR on Linux,
     * and %VAR% on Windows.
     */
    public static String $$(String variable) {
      return PARAMETER_EXPANSION_LEFT + variable + PARAMETER_EXPANSION_RIGHT;
    }
    
    /**
     * This constant is used to construct class path and it will be replaced with
     * real class path separator(':' for Linux and ';' for Windows) by
     * NodeManager on container launch. User has to use this constant to construct
     * class path if user wants cross-platform practice i.e. submit an application
     * from a Windows client to a Linux/Unix server or vice versa.
     */
    public static final String CLASS_PATH_SEPARATOR= "<CPS>";
    
    /**
     * Default platform-agnostic CLASSPATH for YARN applications. A
     * comma-separated list of CLASSPATH entries. The parameter expansion marker
     * will be replaced with real parameter expansion marker ('%' for Windows and
     * '$' for Linux) by NodeManager on container launch. For example: {{VAR}}
     * will be replaced as $VAR on Linux, and %VAR% on Windows.
     */
    public static final String[] DEFAULT_YARN_CROSS_PLATFORM_APPLICATION_CLASSPATH= {
            $$(ApplicationConstants.Environment.HADOOP_CONF_DIR.toString()),
            $$(ApplicationConstants.Environment.HADOOP_COMMON_HOME.toString())
                + "/share/hadoop/common/*",
            $$(ApplicationConstants.Environment.HADOOP_COMMON_HOME.toString())
                + "/share/hadoop/common/lib/*",
            $$(ApplicationConstants.Environment.HADOOP_HDFS_HOME.toString())
                + "/share/hadoop/hdfs/*",
            $$(ApplicationConstants.Environment.HADOOP_HDFS_HOME.toString())
                + "/share/hadoop/hdfs/lib/*",
            $$(ApplicationConstants.Environment.HADOOP_YARN_HOME.toString())
                + "/share/hadoop/yarn/*",
            $$(ApplicationConstants.Environment.HADOOP_YARN_HOME.toString())
                + "/share/hadoop/yarn/lib/*" 
     };
    

    public static void createDir(FileSystem fs, Path dir, FsPermission permission) {
        String warningMsg;
        try {
            if(!fs.exists(dir)) {
                fs.mkdirs(dir);
                fs.setPermission(dir, permission);
                return;
            }
            warningMsg = "Directory " + dir + " already exists!";
            LOG.info(warningMsg);
        } catch (IOException e) {
            warningMsg = "Failed to create " + dir + ": " + e.toString();
            LOG.error(warningMsg);
        }
    }
    
    public static void givePerms(FileSystem fs, File target, boolean recursive) throws IllegalArgumentException, IOException {
        fs.setPermission(new Path(target.getPath()),  new FsPermission((short) 0777));
        LOG.info(target.getPath());
        if (target.isDirectory() && recursive) {
            File[] files = target.listFiles();
            for (File file : files) {
                givePerms(fs, file, recursive);
            }
        }
    }
    

    /**
     * @param fs 
     *        FileSystem, 
     *        paths
     *        File paths in hdfs
     * @return
     *        total line number of all these files.
     * @throws IOException 
     */
    public static long getFileLineCount(FileSystem fs, String paths) throws IOException {
        //TODO We could get line count from column config
        long total = 0L;
        if (StringUtils.isNotBlank(paths)) {
            for (String path: paths.split(",")) {
                Path p = new Path(path);
                if (fs.exists(p)) {
                    FSDataInputStream in = fs.open(p);
                    if (path.endsWith(".gz")) {
                        // If this is a compressed file
                        GZIPInputStream gzis = new GZIPInputStream(in);
                        byte[] buffer = new byte[1024];
                        while(gzis.read(buffer) != -1) {
                            String record = new String(buffer);
                            int c = StringUtils.countMatches(record, StringUtils.LF);
                            buffer = new byte[1024];
                            total += c;
                        }
                    } else {
                        BufferedReader d = new BufferedReader(new InputStreamReader(in));
                        String line;
                        while((line = d.readLine()) != null) {
                            if (StringUtils.isNotBlank(line)) {
                                total += 1;
                            }
                        }
                    }
                }
            }
        }
        LOG.info("File line count: " + total);
        return total;
    }
}
