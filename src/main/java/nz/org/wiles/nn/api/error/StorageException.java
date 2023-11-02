package nz.org.wiles.nn.api.error;

public class StorageException  extends RuntimeException {

  public StorageException(String message) {
    super(message);
  }

  public StorageException(String message, Throwable cause) {
    super(message, cause);
  }
}