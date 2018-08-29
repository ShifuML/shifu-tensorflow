package ml.shifu.shifu.tensorflow;

import org.junit.Test;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import ml.shifu.shifu.container.obj.GenericModelConfig;

import java.util.Map;
import java.util.HashMap;
import java.util.Random;

public class TensorflowModelTest {

    @Test
    public void testCompute() {
        Map<String, Object> properties = new HashMap<String, Object>();         
        properties.put("tags", new String[]{"serve"});
        properties.put("inputnames", new String[]{"dense_46_input", "dropout_1/keras_learning_phase"});
        properties.put("outputnames", new String[]{"dense_66/Sigmoid"});
        properties.put("dropout_1/keras_learning_phase", false);
        properties.put("modelpath",
            "/Users/wzhu1/workspace/shifu-tensorflow/shifu-tensorflow-eval/src/test/resource/dummydl");

        GenericModelConfig config = new GenericModelConfig();
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
