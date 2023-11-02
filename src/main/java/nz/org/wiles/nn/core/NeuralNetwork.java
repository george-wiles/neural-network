package nz.org.wiles.nn.core;

import nz.org.wiles.nn.model.NeuralNetDataConfig;

import java.util.Arrays;
import java.util.OptionalDouble;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;
    private final double[] bias;
    private final boolean ignoreBias;

    class BackPropagationDelta {
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

    /**
     * Sets up the input, hidden and output layers including initial weights and bias.
     */
    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs,
                         double[][] initial_hidden_layer_weights,
                         double[][] initial_output_layer_weights,
                         double [] bias,
                         double learning_rate) {
        //Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;
        this.bias = bias;
        this.ignoreBias = (bias == null);

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
    }

    public NeuralNetwork(NeuralNetDataConfig config) {
        //Initialise the network
        this.num_inputs = config.getNumInput();
        this.num_hidden = config.getNumHidden();
        this.num_outputs = config.getNumOutput();
        this.learning_rate = config.getLearningRate();

        this.bias = config.getBias();
        this.ignoreBias = (bias == null);

        this.hidden_layer_weights = config.getStartHiddenLayerWeights();
        this.output_layer_weights = config.getStartOutputLayerWeights();


    }


    //Calculate neuron activation for an input
    public double sigmoid(double weighted_sum_for_node) {
        return 1 / (1 + Math.exp(-1 * weighted_sum_for_node));
    }

    //Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs, boolean verbose) {
        if (verbose) System.out.println("------------------------------");
        if (verbose) System.out.println(" Feed forward pass");
        if (verbose) System.out.println("------------------------------");
        if (verbose) System.out.println("inputs: " + Arrays.toString(inputs));
        if (verbose) System.out.println("bias nodes: " + (ignoreBias ? "no bias" : Arrays.toString(bias)));
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            // Calculate the weighted sum, and then compute the final output.
            double weighted_sum_for_node = 0.0;
            if (verbose) System.out.print("Z" + (5+i) + "=");
            for (int j = 0; j < hidden_layer_weights.length; j++) {
                if (verbose) System.out.printf("(%f * %f) + ", hidden_layer_weights[j][i], inputs[j] );
                weighted_sum_for_node += (hidden_layer_weights[j][i] * inputs[j]);
            }
            // bias nodes dimensions exist for hidden layer + output layer.
            // this index start from 0 to length of hidden layer
            double b = (ignoreBias) ? 0.00 : bias[i];
            weighted_sum_for_node += b;
            if (verbose) System.out.printf("%f = %f%n", b, weighted_sum_for_node);
            double output = sigmoid(weighted_sum_for_node);
            if (verbose) System.out.println("O" + (5+i) + "=" + output);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            // Calculate the weighted sum, and then compute the final output.
            double weighted_sum_for_node = 0.0;
            if (verbose) System.out.print("Z" + (7+i) + "=");
            for (int j = 0; j < output_layer_weights.length; j++) {
                if (verbose) System.out.print(
                        String.format("(%f * %f) + ", output_layer_weights[j][i], hidden_layer_outputs[j] )
                );
                weighted_sum_for_node += (output_layer_weights[j][i] * hidden_layer_outputs[j]);
            }
            // bias nodes dimensions exist for [hidden layer + output layer].
            // this index start from the beginning of output layer biases.
            // bias[hidden.length + output.length = [hb1, hb2, ob1, ob2, ob3]
            double b = ignoreBias ? 0.00 : bias[num_hidden + i];
            weighted_sum_for_node += b;
            if (verbose) System.out.println(String.format("%f = %f", b, weighted_sum_for_node));
            double output = sigmoid(weighted_sum_for_node);
            if (verbose) System.out.println("O" + (7+i) + "=" + output);
            output_layer_outputs[i] = output;
        }
        if (verbose) System.out.println("HL outputs: " + Arrays.toString(hidden_layer_outputs));
        if (verbose) System.out.println("OL outputs: " + Arrays.toString(output_layer_outputs));
        return new double[][]{hidden_layer_outputs, output_layer_outputs};
    }

    public BackPropagationDelta backward_propagate_error(double[] inputs,
                                                         double[] hidden_layer_outputs,
                                                 double[] output_layer_outputs, int desired_outputs, boolean verbose) {

        if (verbose) System.out.println("------------------------------");
        if (verbose) System.out.println(" Back Propagation");
        if (verbose) System.out.println("------------------------------");
        int[] d = new int[num_outputs];
        d[desired_outputs] = 1;

        double[] output_layer_betas = new double[num_outputs];
        // Calculate output layer betas.
        //  Compute 𝛽𝑧 = 𝑑𝑧 − 𝑜𝑧 --> for each output node
        for (int i = 0; i < num_outputs; i++) {
            double bi = (double)d[i] - output_layer_outputs[i];
            if (verbose) System.out.println("b" + (7+i) + "=" + d[i] + "-" + output_layer_outputs[i] + " => " + bi);
            output_layer_betas[i] = bi;
        }
        if (verbose) System.out.println("OL betas: " + Arrays.toString(output_layer_betas));

        double[] hidden_layer_betas = new double[num_hidden];
        // Compute 𝛽𝑗 = σ𝑘 𝑤𝑗→𝑘𝑜𝑘 1 − 𝑜𝑘 𝛽𝑘 for each hidden node
        for (int i = 0; i < num_hidden; i++) {
            double btotal = 0.0;
            if (verbose) System.out.print("b" + (5+i) + "=");
            for (int k = 0; k < num_outputs; k++) {
                if (verbose) System.out.println(
                        String.format("(%f * %f * (1 - %f) * %f) +", output_layer_weights[i][k], output_layer_outputs[k], output_layer_outputs[k], output_layer_betas[k] )
                );
                btotal += output_layer_weights[i][k] * output_layer_outputs[k] * (1 - output_layer_outputs[k]) * output_layer_betas[k];
            }
            if (verbose) System.out.println(" --> " + btotal);
            hidden_layer_betas[i] = btotal;
        }
        if (verbose) System.out.println("HL betas: " + Arrays.toString(hidden_layer_betas));

        // This is a HxO array (H hidden nodes, O outputs)
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];

        // Calculate output layer weight change
        for (int j = 0; j < num_hidden; j++) {
            for (int k = 0; k < num_outputs; k++) {
                if (verbose) {
                    System.out.println(
                            String.format("w%d%d = %f * %f * %f * (1 - %f) * %f", (5+j), (7+k), learning_rate, hidden_layer_outputs[j], output_layer_outputs[k], output_layer_outputs[k], output_layer_betas[k] )
                    );
                }
                delta_output_layer_weights[j][k] = learning_rate * hidden_layer_outputs[j] * output_layer_outputs[k] * (1 - output_layer_outputs[k]) * output_layer_betas[k];
             }
        }

        if (verbose) System.out.println("Delta output weights: " + Arrays.deepToString(delta_output_layer_weights));

        // This is a IxH array (I inputs, H hidden nodes)
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        // Calculate hidden layer weight changes.
        for (int j = 0; j < num_inputs; j++) {
            for (int k = 0; k < num_hidden; k++) {
                if (verbose) {
                    System.out.println(
                            String.format("w%d%d = %f * %f * %f * (1 - %f) * %f",
                                    (1+j), (5+k), learning_rate, inputs[j], hidden_layer_outputs[k], hidden_layer_outputs[k], hidden_layer_betas[k])
                    );
                }
                delta_hidden_layer_weights[j][k] =
                        learning_rate * inputs[j] * hidden_layer_outputs[k] * (1 - hidden_layer_outputs[k]) *  hidden_layer_betas[k];
            }
        }

        double[] delta_hidden_bias = null;
        double[] delta_output_bias = null;
        if (!ignoreBias) {
            delta_hidden_bias = calculateBias(verbose, num_hidden, 5, hidden_layer_outputs, hidden_layer_betas);
            delta_output_bias = calculateBias(verbose, num_outputs, 7, output_layer_outputs, output_layer_betas);
            if (verbose) System.out.println("Bias deltas: " + Arrays.toString(delta_hidden_bias) + Arrays.toString(delta_output_bias));
        }

        if (verbose) System.out.println("Delta hidden weights: " + Arrays.deepToString(delta_hidden_layer_weights));
        // Return the weights we calculated, so they can be used to update all the weights.
        return new BackPropagationDelta(delta_hidden_layer_weights,
                delta_output_layer_weights, delta_hidden_bias, delta_output_bias);
    }

    private double[] calculateBias(boolean verbose, int num_bias, int node, double[] outputs, double[] betas) {
        double[] bias = new double[num_bias];
        for (int i = 0; i < num_bias; i++) {
            if (verbose) System.out.println(
                    String.format("bias%d = %f * %f * (1 - %f) * %f",
                            (node+ i), learning_rate, outputs[i], outputs[i], betas[i])
            );
            bias[i] = learning_rate * outputs[i] * (1 - outputs[i]) * betas[i];
        }
        return bias;
    }


    public void update_weights(BackPropagationDelta bpDelta) {
        // Update the weights
        for (int i = 0; i <hidden_layer_weights.length; i++) {
            for (int j = 0; j < hidden_layer_weights[i].length; j++) {
                hidden_layer_weights[i][j] += bpDelta.delta_hidden_layer_weights[i][j];
            }
        }
        for (int i = 0; i <output_layer_weights.length; i++) {
            for (int j = 0; j < output_layer_weights[i].length; j++) {
                output_layer_weights[i][j] += bpDelta.delta_output_layer_weights[i][j];
            }
        }

        if (!ignoreBias) {
            for (int j = 0; j < num_hidden; j++) {
                bias[j] += bpDelta.delta_hidden_bias[j];
            }

            for (int j = 0; j < num_outputs; j++) {
                bias[num_hidden + j] += bpDelta.delta_output_bias[j];
            }
        }

    }

    public void train(double[][] instances, int[] desired_outputs, int epochs, boolean verbose) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance, verbose);
                BackPropagationDelta bpDelta = backward_propagate_error(instance, outputs[0], outputs[1], desired_outputs[i], verbose);
                int predicted_class = findMaxElement(outputs[1]);
                predictions[i] = predicted_class;

                //We use online learning, i.e. update the weights after every instance.
                update_weights(bpDelta);
            }

            // Print new weights
            if (verbose) System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            if (verbose) System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // Print accuracy achieved over this epoch
            double acc = calculateAccuracy(predictions, desired_outputs);
            System.out.println(String.format("epoch %d: accuracy = %f", epoch, acc));
        }
    }

    public double calculateAccuracy(int[] predictions, int[] desired_outputs) {
        int correct = 0;
        for (int i = 0; i < desired_outputs.length; i++) {
            if (predictions[i] == desired_outputs[i]) {
                correct++;
            }
        }
        System.out.print("accuracy predicted: [" + correct + "/" + desired_outputs.length + "]: ");
        return (double)correct / desired_outputs.length;
    }

    public int[] predict(double[][] instances, boolean verbose) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance, verbose);
            int predicted_class = findMaxElement(outputs[1]);
            predictions[i] = predicted_class;
        }
        return predictions;
    }

    private int findMaxElement(double[] output) {
        OptionalDouble optionalDouble = Arrays.stream(output).max();
        if (optionalDouble.isPresent()) {
            for (int j = 0; j < output.length; j++) {
                if (optionalDouble.getAsDouble() == output[j]) {
                    return j;
                }
            }
        }
        throw new RuntimeException("cannot find max in empty array " + output);
    }

}
