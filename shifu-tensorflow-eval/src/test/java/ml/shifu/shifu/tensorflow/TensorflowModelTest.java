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
package ml.shifu.shifu.tensorflow;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.junit.Test;

import ml.shifu.shifu.container.obj.GenericModelConfig;

public class TensorflowModelTest {

    @Test
    public void testCompute() {
        Map<String, Object> properties = new HashMap<String, Object>();
        properties.put("tags", Arrays.asList("serve"));
        properties.put("inputnames", new String[] { "dense_46_input", "dropout_1/keras_learning_phase" });
        properties.put("outputnames", new String[] { "dense_66/Sigmoid" });
        properties.put("dropout_1/keras_learning_phase", false);
        properties.put("modelpath", System.getProperty("user.dir") + File.separator + "src/test/resources/dummydl");

        GenericModelConfig config = new GenericModelConfig();
        List<String> inputs = new ArrayList<String>();
        inputs.add("dense_46_input");
        inputs.add("dropout_1/keras_learning_phase");
        config.setInputnames(inputs);
        config.setProperties(properties);
        double[] inputArr = new double[1522];
        Random random = new Random();
        for(int i = 0; i < 1522; i++) {
            inputArr[i] = random.nextDouble();
        }
        MLData input = new BasicMLData(inputArr);
        TensorflowModel tm = new TensorflowModel();
        tm.init(config);
        double result = tm.compute(input);
        System.out.println(result);
    }
}
