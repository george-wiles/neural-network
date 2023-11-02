package nz.org.wiles.nn.service;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties("storage")
public class StorageProperties {

  private String location = "upload-dir";
  private String resources = "src/main/resources/data";

  public String getLocation() {
    return location;
  }

  public String getInitialLocation() { return resources; }

}