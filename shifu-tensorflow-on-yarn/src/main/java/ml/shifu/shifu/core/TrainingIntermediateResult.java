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
package ml.shifu.shifu.core;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

/**
 * Here we store workers' intermediate result and sending from worker container to app master
 * 
 * @author webai
 *
 */
public class TrainingIntermediateResult implements Serializable {

    private static final long serialVersionUID = -706025707830446870L;

    private int currentEpochStep = 0;
    private int workerIndex = -1;

    private double trainingError = -1.0d;
    private double validError = -1.0d;

    private double currentEpochTime = 0; // seconds

    private double currentEpochValidTime = 0; // seconds

    public TrainingIntermediateResult() {
    }

    public TrainingIntermediateResult(byte[] result) throws IOException, ClassNotFoundException {
        ByteArrayInputStream in = new ByteArrayInputStream(result);
        ObjectInputStream is = new ObjectInputStream(in);
        TrainingIntermediateResult clone = (TrainingIntermediateResult) is.readObject();
        deepClone(clone);
    }

    public int getCurrentEpochStep() {
        return currentEpochStep;
    }

    public void setCurrentEpochStep(int currentEpochStep) {
        this.currentEpochStep = currentEpochStep;
    }

    public int getWorkerIndex() {
        return workerIndex;
    }

    public void setWorkerIndex(int workerIndex) {
        this.workerIndex = workerIndex;
    }

    public double getTrainingError() {
        return trainingError;
    }

    public void setTrainingError(double trainingError) {
        this.trainingError = trainingError;
    }

    public double getValidError() {
        return validError;
    }

    public void setValidError(double validError) {
        this.validError = validError;
    }

    public double getCurrentEpochTime() {
        return currentEpochTime;
    }

    public void setCurrentEpochTime(double currentEpochTime) {
        this.currentEpochTime = currentEpochTime;
    }

    public byte[] serialize() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ObjectOutputStream os = new ObjectOutputStream(out);
        os.writeObject(this);
        return out.toByteArray();
    }

    public void deepClone(TrainingIntermediateResult clone) {
        if(clone != null) {
            this.setCurrentEpochStep(clone.currentEpochStep);
            this.setCurrentEpochTime(clone.currentEpochTime);
            this.setTrainingError(clone.trainingError);
            this.setValidError(clone.validError);
            this.setWorkerIndex(clone.workerIndex);
            this.setCurrentEpochValidTime(clone.currentEpochValidTime);
        }
    }

    /**
     * @return the currentEpochValidTime
     */
    public double getCurrentEpochValidTime() {
        return currentEpochValidTime;
    }

    /**
     * @param currentEpochValidTime
     *            the currentEpochValidTime to set
     */
    public void setCurrentEpochValidTime(double currentEpochValidTime) {
        this.currentEpochValidTime = currentEpochValidTime;
    }

    @Override
    public String toString() {
        return "TrainingIntermediateResult [currentEpochStep=" + currentEpochStep + ", workerIndex=" + workerIndex
                + ", trainingError=" + trainingError + ", validError=" + validError + ", currentEpochTime="
                + currentEpochTime + ", currentEpochValidTime=" + currentEpochValidTime + "]";
    }

}
