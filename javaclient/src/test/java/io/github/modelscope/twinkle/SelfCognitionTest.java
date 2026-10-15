package io.github.modelscope.twinkle;

import com.google.gson.JsonElement;
import io.github.modelscope.twinkle.model.ModelClient;
import io.github.modelscope.twinkle.processor.DataLoaderClient;
import io.github.modelscope.twinkle.processor.DatasetClient;
import io.github.modelscope.twinkle.types.DatasetKind;
import io.github.modelscope.twinkle.types.DatasetMeta;
import io.github.modelscope.twinkle.types.LoraConfig;
import io.github.modelscope.twinkle.types.SaveResponse;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Runs an end-to-end self-cognition LoRA training example in IntelliJ IDEA.
 *
 * <p>The example mirrors the self-hosted Python workflow.</p>
 */
public final class SelfCognitionTest {

    private static final Logger LOGGER = Logger.getLogger(SelfCognitionTest.class.getName());

    private SelfCognitionTest() {}

    /** Starts an end-to-end remote LoRA training run. */
    public static void main(String[] args) {
        // Step 1: Configure the server, model, dataset, and training hyperparameters.
        String baseUrl = "************";
        String token = "************";
        String modelId = "Qwen/Qwen3.6-27B";
        String datasetId = "************";
        String template = "Qwen3_5Template";
        int batchSize = 4;
        int epochs = 1;
        double learningRate = Double.parseDouble("0.0001");

        // Step 2: Create the client and verify that the Twinkle server is available.
        try (TwinkleClient client =
                TwinkleClient.builder().baseUrl(baseUrl).apiKey(token).build()) {
            if (!client.healthCheck()) {
                throw new IllegalStateException("Twinkle server health check failed");
            }

            LOGGER.log(Level.INFO, "Server capabilities: {0}", client.serverCapabilities());
            LOGGER.log(Level.INFO, "Server capacity: {0}", client.capacityInfo());

            // Step 3: Load, format, and encode the self-cognition dataset on the server.
            DatasetClient dataset =
                    client.processors().dataset(DatasetKind.DATASET, Map.of("dataset_meta", DatasetMeta.of(datasetId)));
            dataset.setTemplate(template, Map.of("model_id", modelId));
            dataset.encode(false, Map.of("batched", true));

            // Step 4: Create a data loader and configure the model, LoRA adapter, and optimizer.
            DataLoaderClient dataLoader =
                    client.processors().dataLoader(dataset.processorId(), Map.of("batch_size", batchSize));
            ModelClient model = client.models().open(modelId);
            model.addAdapter(
                    "default",
                    new LoraConfig(8, 16, "all-linear", 0.01, "none", null),
                    Map.of("gradient_accumulation_steps", 1));
            model.setTemplate(template, Map.of());
            model.setProcessor("InputProcessor", Map.of("padding_side", "right"));
            model.setLoss("CrossEntropyLoss");
            model.setOptimizer("Adam", Map.of("lr", learningRate));

            // Step 5: Run forward/backward passes and apply optimizer steps for each batch.
            for (int epoch = 0; epoch < epochs; epoch++) {
                int step = 0;
                try {
                    for (JsonElement batch : dataLoader) {
                        model.forwardBackward(batch);
                        model.clipGradAndStep(1.0, 2);
                        LOGGER.log(Level.INFO, "Epoch {0}, step {1}, metric: {2}", new Object[] {
                            epoch + 1, step, model.calculateMetric(true)
                        });
                        step++;
                    }
                } catch (NoSuchElementException e) {
                    LOGGER.log(Level.INFO, "Epoch {0} completed with {1} steps", new Object[] {epoch + 1, step});
                    break;
                }
                LOGGER.log(Level.INFO, "Epoch {0} completed with {1} steps", new Object[] {epoch + 1, step});
            }

            // Step 6: Save the final checkpoint on the server.
            SaveResponse saved = model.save("twinkle-java-final", true);
            LOGGER.log(Level.INFO, "Saved checkpoint: {0}", saved.twinklePath());
        }
    }
}
