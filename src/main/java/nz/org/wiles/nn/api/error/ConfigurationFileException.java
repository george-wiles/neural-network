package nz.org.wiles.nn.api.error;

public class ConfigurationFileException extends StorageException {

  public ConfigurationFileException(String message) {
    super(message);
  }

  public ConfigurationFileException(String message, Throwable cause) {
    super(message, cause);
  }
}
