package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.transport.TwinkleSerializable;

/**
 * Configures a LoRA adapter using the Twinkle protocol.
 *
 * @param rank the low-rank dimension
 * @param loraAlpha the LoRA scaling factor
 * @param targetModules the module name or module names to receive LoRA layers
 * @param loraDropout the dropout probability for LoRA layers
 * @param bias the bias training strategy
 * @param taskType an optional downstream task type
 */
public record LoraConfig(
        int rank, int loraAlpha, Object targetModules, double loraDropout, String bias, String taskType)
        implements TwinkleSerializable {
    public LoraConfig {
        if (rank <= 0) {
            throw new IllegalArgumentException("rank must be greater than 0");
        }
    }

    /** Creates a configuration with the same defaults as the Python client. */
    public LoraConfig() {
        this(8, 32, "all-linear", 0.0, "none", null);
    }

    @Override
    public JsonObject toTwinkleJson() {
        JsonObject value = new JsonObject();
        value.addProperty("_TWINKLE_TYPE_", "LoraConfig");
        value.addProperty("r", rank);
        value.addProperty("lora_alpha", loraAlpha);
        value.add("target_modules", new com.google.gson.Gson().toJsonTree(targetModules));
        value.addProperty("lora_dropout", loraDropout);
        value.addProperty("bias", bias);
        if (taskType != null) {
            value.addProperty("task_type", taskType);
        }
        return value;
    }
}
