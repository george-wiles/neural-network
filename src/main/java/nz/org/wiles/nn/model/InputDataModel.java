package nz.org.wiles.nn.model;

public class InputDataModel {
  private final String[] labels;
  private final String[] header;
  private final double[][] data;

  public InputDataModel(String[] labels,
                            String[]  header,
                            double[][] data) {
    this.labels = labels;
    this.header = header;
    this.data = data;

  }

  public String[] getLabels() {
    return labels;
  }

  public String[] getHeader() {
    return header;
  }

  public double[][] getData() {
    return data;
  }
}
