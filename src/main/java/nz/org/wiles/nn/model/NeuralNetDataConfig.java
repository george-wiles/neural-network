package nz.org.wiles.nn.model;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Neural net data config to match input data.
 */
@ConfigurationProperties("nn-config")
public class NeuralNetDataConfig {
  private final double learningRate;
  private final int numInput;
  private final int numHidden;
  private final int numOutput;

  private final double[][] startHiddenLayerWeights;
  private final double[][] startOutputLayerWeights;

  public double[] getBias() {
    return bias;
  }

  private final double[] bias;


  public NeuralNetDataConfig(double learningRate,
                             int numInput,
                             int numHidden,
                             int numOutput,
                             double[][] startHiddenLayerWeights,
                             double[][] startOutputLayerWeights,
                             double[] bias) {
    this.learningRate = learningRate;
    this.numInput = numInput;
    this.numHidden = numHidden;
    this.numOutput = numOutput;
    this.startHiddenLayerWeights = startHiddenLayerWeights;
    this.startOutputLayerWeights =  startOutputLayerWeights;
    this.bias = bias;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public int getNumInput() {
    return numInput;
  }

  public int getNumHidden() {
    return numHidden;
  }

  public int getNumOutput() {
    return numOutput;
  }

  public double[][] getStartHiddenLayerWeights() {
    return startHiddenLayerWeights;
  }

  public double[][] getStartOutputLayerWeights() {
    return startOutputLayerWeights;
  }
}


