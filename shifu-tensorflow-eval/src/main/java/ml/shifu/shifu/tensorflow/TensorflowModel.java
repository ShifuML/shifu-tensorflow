package ml.shifu.shifu.tensorflow;

import org.encog.ml.data.MLData;
import ml.shifu.shifu.core.Computable;
import ml.shifu.shifu.container.obj.GenericModelConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.Map;
import java.util.HashMap;
import java.util.List;

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

    @Override
    public double compute(MLData input) {
        LOG.error("tensorflow compute start.");
        double result = Double.MIN_VALUE;
        if(initiate && smb != null) {
            Session.Runner runner = smb.session().runner();
            double[] inputDArr = input.getData();
            float[] inputFArr = new float[inputDArr.length];
            for(int i = 0; i < inputDArr.length; i++) {
                inputFArr[i] = (float)inputDArr[i];
            }
            
            runner.feed(inputNames[0], Tensor.create(new float[][]{inputFArr}));

            for(int i = 1; i < inputNames.length; i++) {
                try {
                    runner.feed(inputNames[i], Tensor.create(properties.get(inputNames[i])));
                } catch(Exception e) {
                    LOG.error("Invalid input, {}", e);
                }
            }

            runner.fetch(outputNames);

            Tensor<?> output = runner.run().get(0);
            result = ((float[][]) output.copyTo(new float[1][1]))[0][0];
        }
        LOG.error("return result {}", result);
        return result;
    }

    @Override
    public void init(GenericModelConfig config) {
        LOG.error("init tensorflow model");
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
        this.modelPath = (String)properties.get("modelpath");
        this.inputNames = config.getInputnames().toArray(new String[config.getInputnames().size()]);
        this.outputNames = (String)properties.get("outputnames");
        this.tags = ((List<String>)properties.get("tags")).toArray(new String[((List<String>)properties.get("tags")).size()]);

        LOG.error("Properties : {}", properties);

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

        LOG.error("Load model from " + this.modelPath);
        this.smb = SavedModelBundle.load(modelPath, tags);
        LOG.error("init tensorflow model done");
        initiate = true;
    }

    @Override
    public void releaseResource() {

    }
}
