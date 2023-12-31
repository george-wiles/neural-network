package nz.org.wiles.nn.api.upload;

import java.io.IOException;
import java.util.stream.Collectors;

import nz.org.wiles.nn.api.error.StorageException;
import nz.org.wiles.nn.service.StorageService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.MvcUriComponentsBuilder;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
@RequestMapping("/api/upload")
public class DataUploadController {
  private final StorageService storageService;

  @Autowired
  public DataUploadController(StorageService storageService) {
    this.storageService = storageService;
  }

  @GetMapping("")
  public String listUploadedFiles(Model model) throws IOException {

    model.addAttribute("files", storageService.loadAll().map(
            path -> MvcUriComponentsBuilder.fromMethodName(DataUploadController.class,
                "serveFile", path.getFileName().toString()).build().toUri().toString())
                                    .collect(Collectors.toList()));

    return "uploadForm";
  }

  @GetMapping("/files/{filename:.+}")
  @ResponseBody
  public ResponseEntity<Resource> serveFile(@PathVariable String filename) {

    Resource file = storageService.loadAsResource(filename);

    if (file == null)
      return ResponseEntity.notFound().build();

    return ResponseEntity.ok().header(HttpHeaders.CONTENT_DISPOSITION,
        "attachment; filename=\"" + file.getFilename() + "\"").body(file);
  }

  @PostMapping("")
  public String handleFileUpload(@RequestParam("file") MultipartFile file,
                                 RedirectAttributes redirectAttributes) {

    storageService.store(file);
    redirectAttributes.addFlashAttribute("message",
        "You successfully uploaded " + file.getOriginalFilename() + "!");

    return "redirect:/";
  }

  @ExceptionHandler(StorageException.class)
  public ResponseEntity<?> handleStorageFileNotFound(StorageException exc) {
    return ResponseEntity.notFound().build();
  }

}
