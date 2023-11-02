package nz.org.wiles.nn.model;

public class BackPropagationDelta {
  final double[][] delta_hidden_layer_weights;
  final double[][] delta_output_layer_weights;
  final double[] delta_hidden_bias;
  final double[] delta_output_bias;

  public BackPropagationDelta(double[][] delta_hidden_layer_weights,
                              double[][] delta_output_layer_weights,
                              double[] delta_hidden_bias,
                              double[] delta_output_bias) {
    this.delta_hidden_layer_weights = delta_hidden_layer_weights;
    this.delta_output_layer_weights = delta_output_layer_weights;
    this.delta_hidden_bias = delta_hidden_bias;
    this.delta_output_bias = delta_output_bias;
  }
}
