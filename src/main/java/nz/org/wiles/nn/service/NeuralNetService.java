package nz.org.wiles.nn.service;

import nz.org.wiles.nn.api.error.PredictionException;
import nz.org.wiles.nn.model.NeuralNetDataConfig;
import nz.org.wiles.nn.model.InputDataModel;
import nz.org.wiles.nn.core.LabelEncoder;
import nz.org.wiles.nn.core.NeuralNetwork;
import nz.org.wiles.nn.core.Rescaler;
import nz.org.wiles.nn.core.Util;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

@Service
public class NeuralNetService {

  private void train(String trainingDataPath, NeuralNetwork neuralNetwork) {
    final InputDataModel trainingModel = getDataArray(trainingDataPath);

    // scale features to [0,1] to improve training
    Rescaler rescaler = new Rescaler(trainingModel.getData());
    rescaler.rescaleData(trainingModel.getData());

    // We can"t use strings as labels directly in the network, so need to do some transformations
    LabelEncoder label_encoder = new LabelEncoder(trainingModel.getLabels());
    // encode "Adelie" as 1, "Gentoo" as 2, "Chinstrap" as 3,...
    int[] integer_encoded = label_encoder.intEncode(trainingModel.getLabels());
    // encode 1 as [1, 0, 0], 2 as [0, 1, 0], and 3 as [0, 0, 1] (to fit with our network outputs!)
    int[][] one_hot_encoded = label_encoder.oneHotEncode(trainingModel.getLabels());

    System.out.printf("First instance has label %s, which is %d as an integer, and %s as a list of outputs.\n",
        trainingModel.getLabels()[0],
        integer_encoded[0],
        Arrays.toString(one_hot_encoded[0]));

    // need to wrap it into a 2D array
    int[] instance1_prediction = neuralNetwork.predict(
        new double[][]{trainingModel.getData()[0]}, true);

    String instance1_predicted_label;
    if (instance1_prediction[0] == -1) {
      throw new PredictionException("Initial Prediction for Penguin is unknown");
    } else {
      instance1_predicted_label = label_encoder.inverse_transform(instance1_prediction[0]);
    }
    System.out.println("Predicted label for the first instance is: " + instance1_predicted_label);

    neuralNetwork.train(new double[][]{trainingModel.getData()[0]}, instance1_prediction, 1, true);

    System.out.println("Weights after performing BP for first instance only:");
    System.out.println("Hidden layer weights:\n" + Arrays.deepToString(neuralNetwork.hidden_layer_weights));
    System.out.println("Output layer weights:\n" + Arrays.deepToString(neuralNetwork.output_layer_weights));

    // Train for 100 epochs, on all instances.
    neuralNetwork.train(trainingModel.getData(), integer_encoded, 100, false);
    System.out.println("Hidden layer weights:\n" + Arrays.deepToString(neuralNetwork.hidden_layer_weights));
    System.out.println("Output layer weights:\n" + Arrays.deepToString(neuralNetwork.output_layer_weights));

  }

  private InputDataModel getDataArray(String trainingDataPath) {
    List<String[]> lines = Util.getLines(trainingDataPath);
    String[] header = lines.remove(0);
    String[] labels = Util.getLabels(lines);
    return new InputDataModel(labels, header, Util.getData(lines));
  }

  private void predict(String testDataPath, NeuralNetwork neuralNetwork) {
    final InputDataModel testModel = getDataArray(testDataPath);
    double[][] instances_test = testModel.getData();
    LabelEncoder label_encoder = new LabelEncoder(testModel.getLabels());
    int[] integer_encoded_test = label_encoder.intEncode(testModel.getLabels());

    // scale the test according to our training data.
    Rescaler rescaler = new Rescaler(testModel.getData());
    rescaler.rescaleData(instances_test);

    // Compute and print the test accuracy
    int[] test_predictions = neuralNetwork.predict(instances_test, false);

    double acc = neuralNetwork.calculateAccuracy(test_predictions, integer_encoded_test);
    System.out.println(String.format("Classify: testset [%d] rows: accuracy = %f", instances_test.length, acc));

  }

  public void run(Resource testingData, Resource trainingData, NeuralNetDataConfig config) throws IOException {
    NeuralNetwork neuralNetwork = new NeuralNetwork(config);
    train(trainingData.getFile().getPath(), neuralNetwork);
    predict(testingData.getFile().getPath(), neuralNetwork);
  }
}
