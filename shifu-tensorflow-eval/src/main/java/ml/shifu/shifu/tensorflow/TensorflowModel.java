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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.encog.ml.data.MLData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import ml.shifu.shifu.container.obj.GenericModelConfig;
import ml.shifu.shifu.core.Computable;

public class TensorflowModel implements Computable {

    private final static Logger LOG = LoggerFactory.getLogger(TensorflowModel.class);

    public Map<String, Object> properties = new HashMap<String, Object>();

    private boolean initiate = false;

    private String modelPath;

    private SavedModelBundle smb;

    private GenericModelConfig config;

    private String[] tags;

    private String[] inputNames;

    private String outputNames;

    /**
     * If enable keras learning phase in training, such bool type of operations would be set as placeholder parameters
     * and in eval should be set to true;
     */
    private List<String> kerasLearningPhaseTensors;

    @Override
    public double compute(MLData input) {
        double result = Double.MIN_VALUE;
        if(!initiate || smb == null) {
            throw new IllegalStateException("TF model not initialized.");
        }

        Tensor<?>[] propertyTensors = null;
        List<Tensor<?>> results = null;
        List<Tensor<?>> tlpTensors = null;

        try {
            Session.Runner runner = smb.session().runner();
            double[] inputDArr = input.getData();
            float[] inputFArr = new float[inputDArr.length];
            for(int i = 0; i < inputDArr.length; i++) {
                inputFArr[i] = (float) inputDArr[i];
            }

            Tensor<?> inputTensor = Tensor.create(new float[][] { inputFArr });
            runner.feed(inputNames[0], inputTensor);

            if(kerasLearningPhaseTensors != null && kerasLearningPhaseTensors.size() > 0) {
                tlpTensors = new ArrayList<>();
                for(String klpTensor: kerasLearningPhaseTensors) {
                    Tensor<?> tensor = Tensor.create(Boolean.FALSE, Boolean.class);
                    tlpTensors.add(tensor);
                    runner.feed(klpTensor, tensor);
                }
            }

            propertyTensors = new Tensor[inputNames.length];
            propertyTensors[0] = inputTensor;
            for(int i = 1; i < inputNames.length; i++) {
                try {
                    propertyTensors[i] = Tensor.create(properties.get(inputNames[i]));
                    runner.feed(inputNames[i], propertyTensors[i]);
                } catch (Exception e) {
                    LOG.error("Invalid input, {}", e);
                }
            }

            runner.fetch(outputNames);
            results = runner.run();
            Tensor<?> output = results.get(0);
            result = ((float[][]) output.copyTo(new float[1][1]))[0][0];
        } catch (Exception e) {
            LOG.error("Error in model inference:", e);
        } finally {
            closeTensors(propertyTensors, results, tlpTensors);
        }

        return result;
    }

    @SuppressWarnings("rawtypes")
    private void closeTensors(Tensor[] propertyTensors, List<Tensor<?>> results, List<Tensor<?>> tlpTensors) {
        if(propertyTensors != null) {
            for(Tensor<?> tensor: propertyTensors) {
                if(tensor != null) {
                    tensor.close();
                }
            }
        }

        if(results != null) {
            for(Tensor<?> tensor: results) {
                if(tensor != null) {
                    tensor.close();
                }
            }
        }

        if(tlpTensors != null) {
            for(Tensor<?> tensor: tlpTensors) {
                if(tensor != null) {
                    tensor.close();
                }
            }
        }
    }

    @Override
    public void init(GenericModelConfig config) {
        LOG.info("Init tensorflow model");
        if(this.initiate) {
            return;
        }
        if(config == null) {
            LOG.error("Config is null");
            throw new RuntimeException("Config is null");
        }

        this.config = config;
        properties = this.config.getProperties();
        if(properties == null || properties.size() == 0) {
            LOG.error("Properties is null");
            throw new RuntimeException("Properties is null");
        }
        this.modelPath = (String) properties.get("modelpath");
        this.inputNames = config.getInputnames().toArray(new String[0]);
        Object outputNames = properties.get("outputnames");
        if(outputNames instanceof String) {
            this.outputNames = (String) properties.get("outputnames");
        } else if(outputNames instanceof String[]) {
            String[] outputs = (String[]) outputNames;
            if(outputs.length == 1) {
                this.outputNames = outputs[0];
            } else {
                throw new IllegalArgumentException("Output now only support single output in inference.");
            }
        }

        @SuppressWarnings("unchecked")
        List<String> tagList = (List<String>) properties.get("tags");
        this.tags = tagList.toArray(new String[tagList.size()]);

        LOG.info("Debug: properties : {}", properties);

        if(this.modelPath == null || this.modelPath.isEmpty()) {
            LOG.error("Model path is null");
            throw new RuntimeException("Model path is null");
        }

        if(this.inputNames == null || this.inputNames.length == 0) {
            LOG.error("Input names is null");
            throw new RuntimeException("Input names is null");
        }

        if(this.outputNames == null || this.outputNames.isEmpty()) {
            LOG.error("Output names is null");
            throw new RuntimeException("Output names is null");
        }

        if(this.tags == null || this.tags.length == 0) {
            LOG.error("Tags is null");
            throw new RuntimeException("Tags is null");
        }

        LOG.info("Load model from {}.", this.modelPath);
        this.smb = SavedModelBundle.load(modelPath, this.tags);

        Iterator<Operation> operations = this.smb.graph().operations();

        kerasLearningPhaseTensors = new ArrayList<String>();
        while(operations.hasNext()) {
            Operation operation = operations.next();
            if("placeholder".equals(operation.type().toLowerCase()) && operation.name().contains("keras_learning_phase")) {
                kerasLearningPhaseTensors.add(operation.name());
            }
        }
        LOG.info("DEBUG: Keras learning phase tensors: {}.", kerasLearningPhaseTensors);

        LOG.info("Init tensorflow model done.");
        initiate = true;
    }

    @Override
    public void releaseResource() {
    }
}
