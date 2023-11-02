package nz.org.wiles.nn;

import nz.org.wiles.nn.model.NeuralNetDataConfig;
import nz.org.wiles.nn.service.StorageProperties;
import nz.org.wiles.nn.service.StorageService;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
@EnableConfigurationProperties({StorageProperties.class, NeuralNetDataConfig.class})
public class NNApplication {

	public static void main(String[] args) {
		SpringApplication.run(NNApplication.class, args);
	}

	@Bean
	CommandLineRunner init(StorageService storageService) {
		return (args) -> {
			storageService.deleteAll();
			storageService.init();
		};
	}

}
