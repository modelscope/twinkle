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
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Runs the Java translation of the self-hosted multimodal LoRA training example.
 *
 * <p>The configured JSONL dataset must already contain model-ready multimodal samples. It is read
 * directly by the Twinkle server; Java only orchestrates the remote workflow.
 */
public final class MultiModalTrainingTest {

    private static final String TEMPLATE = "Qwen3_5Template";
    private static final Logger LOGGER = Logger.getLogger(MultiModalTrainingTest.class.getName());

    private MultiModalTrainingTest() {}

    /** Starts an end-to-end multimodal LoRA training run. */
    public static void main(String[] args) {

        // Step 1: Configure the server, model, dataset, and training hyperparameters.
        String baseUrl = "************";
        String token = "************";
        String modelId = "Qwen/Qwen3.5-4B";
        String datasetId = "/************";
        String template = "Qwen3_5Template";

        int sampleLimit = 20;
        int batchSize = 4;
        int epochs = 1;
        double learningRate = Double.parseDouble("0.0001");

        // Step 2: Create the client that communicates with the Twinkle server.
        try (TwinkleClient client =
                TwinkleClient.builder().baseUrl(baseUrl).apiKey(token).build()) {

            // Step 3: Load the prepared multimodal dataset and create a batch data loader.
            DatasetMeta datasetMeta =
                    new DatasetMeta(datasetId, "default", "train", DatasetMeta.range(0, sampleLimit, 1), null);
            DatasetClient dataset =
                    client.processors().dataset(DatasetKind.LAZY_DATASET, Map.of("dataset_meta", datasetMeta));

            DataLoaderClient dataLoader =
                    client.processors().dataLoader(dataset.processorId(), Map.of("batch_size", batchSize));

            // Step 4: Configure the model, LoRA adapter, loss function, and optimizer.
            ModelClient model = client.models().open(modelId);
            model.addAdapter("default", new LoraConfig(), Map.of("gradient_accumulation_steps", 2));
            model.setTemplate(TEMPLATE, Map.of());
            model.setProcessor("InputProcessor", Map.of("padding_side", "right"));
            model.setLoss("CrossEntropyLoss");
            model.setOptimizer("Adam", Map.of("lr", learningRate));

            // Step 5: Train the model one batch at a time and report metrics periodically.
            for (int epoch = 0; epoch < epochs; epoch++) {
                int step = 0;
                for (JsonElement batch : dataLoader) {
                    model.forwardBackward(batch);
                    model.clipGradAndStep(1.0, 2);
                    if (step % 2 == 0) {
                        LOGGER.log(Level.INFO, "Epoch {0}, step {1}, metric: {2}", new Object[] {
                            epoch + 1, step, model.calculateMetric(true)
                        });
                    }
                    step++;
                }

                // Step 6: Save a checkpoint after each completed epoch.
                SaveResponse checkpoint = model.save("twinkle-multimodal-epoch-" + epoch, true);
                LOGGER.log(Level.INFO, "Saved checkpoint: {0}", checkpoint.twinklePath());
            }
        }
    }
}
