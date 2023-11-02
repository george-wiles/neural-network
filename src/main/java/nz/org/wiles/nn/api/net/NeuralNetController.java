package nz.org.wiles.nn.api.net;

import nz.org.wiles.nn.model.NeuralNetDataConfig;
import nz.org.wiles.nn.service.NeuralNetService;
import nz.org.wiles.nn.service.StorageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;

import java.io.IOException;

@Controller
@RequestMapping("/api/net")
public class NeuralNetController {

  private final StorageService storageService;

  private final NeuralNetDataConfig neuralNetDataConfig;

  private final NeuralNetService neuralNetService;

  @Autowired
  public NeuralNetController(StorageService storageService,
                             NeuralNetDataConfig neuralNetDataConfig,
                             NeuralNetService neuralNetService) {
    this.storageService = storageService;
    this.neuralNetService = neuralNetService;
    this.neuralNetDataConfig = neuralNetDataConfig;
  }

  @GetMapping("")
  public String run(Model model) throws IOException {

    final Resource testData = this.storageService.loadAsResource("penguins.test.csv");
    final Resource trainData = this.storageService.loadAsResource("penguins.train.csv");

    this.neuralNetService.run(testData, trainData, neuralNetDataConfig);

    model.addAttribute("test", testData);
    model.addAttribute("train", trainData);
    return "results";
  }


}
