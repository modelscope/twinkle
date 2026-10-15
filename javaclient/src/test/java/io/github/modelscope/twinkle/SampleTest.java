package io.github.modelscope.twinkle;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.sampler.SamplerClient;
import io.github.modelscope.twinkle.types.SampleRequest;
import io.github.modelscope.twinkle.types.SampleResult;
import io.github.modelscope.twinkle.types.SampledSequence;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Runs the Java translation of the self-hosted text generation example.
 *
 * <p>The sampler and tokenizer run on the Twinkle server. The server returns decoded text in each sampled sequence.</p>
 */
public final class SampleInferenceTest {

    private static final Logger LOGGER = Logger.getLogger(SampleInferenceTest.class.getName());

    private SampleInferenceTest() {}

    /** Starts a remote text generation run. */
    public static void main(String[] args) {
        // Step 1: Configure the server, model, optional LoRA adapter, and sampling settings.
        String baseUrl = "************";
        String apiKey = "************";
        String modelId = "Qwen/Qwen3.5-4B";
        String adapterUri = null;
        int promptCount = 4;
        int samplesPerPrompt = 2;

        // Step 2: Create the client that communicates with the Twinkle server.
        try (TwinkleClient client =winkleClient.builder().baseUrl(baseUrl).apiKey(apiKey).build()) {
            if (!client.healthCheck()) {
                throw new IllegalStateException("Twinkle server health check failed");
            }

            // Step 3: Create a remote sampler for the configured base model.
            SamplerClient sampler = client.samplers().open(modelId);

            // Step 4: Set the chat template used to encode message-based prompts.
            sampler.setTemplate("Qwen3_5Template", null, Map.of("model_id", modelId));

            // Step 5: Prepare one conversation and repeat it for the requested prompt count.
            JsonObject trajectory = conversation("You are a helpful assistant.", "Who are you?");
            List<JsonElement> inputs = Collections.nCopies(promptCount, trajectory);

            // Step 6: Configure generation length, randomness, and candidates per prompt.
            Map<String, Object> samplingParams =
                    Map.of("max_tokens", 128, "temperature", 1.0, "num_samples", samplesPerPrompt);
            SampleRequest request = new SampleRequest(inputs, samplingParams, "", adapterUri, samplesPerPrompt);

            // Step 7: Generate responses on the server, optionally using the configured LoRA adapter.
            List<SampleResult> responses = sampler.sample(request);

            // Step 8: Print the decoded candidates returned by the server.
            LOGGER.log(Level.INFO, "Generated {0} prompts with {1} candidates per prompt", new Object[] {
                responses.size(), samplesPerPrompt
            });
            for (int promptIndex = 0; promptIndex < responses.size(); promptIndex++) {
                SampleResult response = responses.get(promptIndex);
                for (int sequenceIndex = 0; sequenceIndex < response.sequences().size(); sequenceIndex++) {
                    SampledSequence sequence = response.sequences().get(sequenceIndex);
                    LOGGER.log(Level.INFO, "Prompt {0}, sequence {1}: {2}", new Object[] {
                        promptIndex + 1, sequenceIndex + 1, sequence.decoded()
                    });
                }
            }
        }
    }

    private static JsonObject conversation(String systemContent, String userContent) {
        JsonArray messages = new JsonArray();
        messages.add(message("system", systemContent));
        messages.add(message("user", userContent));

        JsonObject trajectory = new JsonObject();
        trajectory.add("messages", messages);
        return trajectory;
    }

    private static JsonObject message(String role, String content) {
        JsonObject message = new JsonObject();
        message.addProperty("role", role);
        message.addProperty("content", content);
        return message;
    }
}
