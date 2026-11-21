package io.github.modelscope.twinkle.internal;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.exception.TwinkleProtocolException;
import io.github.modelscope.twinkle.types.CapacityInfo;
import io.github.modelscope.twinkle.types.CheckpointPath;
import io.github.modelscope.twinkle.types.DeleteCheckpointResult;
import io.github.modelscope.twinkle.types.ServerCapabilities;
import io.github.modelscope.twinkle.types.SupportedModel;
import io.github.modelscope.twinkle.types.WeightsInfo;
import java.util.ArrayList;
import java.util.List;

/** Maps server JSON responses to public records while retaining unknown fields. */
public final class ResponseMapper {

    private ResponseMapper() {}

    public static CapacityInfo capacityInfo(JsonObject source) {
        JsonObject copy = source.deepCopy();
        return new CapacityInfo(
                requiredInt(copy, "max_loras"), requiredInt(copy, "used_loras"), requiredInt(copy, "free_loras"), copy);
    }

    public static ServerCapabilities serverCapabilities(JsonObject source) {
        JsonObject copy = source.deepCopy();
        JsonArray values = required(copy, "supported_models").getAsJsonArray();
        List<SupportedModel> models = new ArrayList<>();

        for (JsonElement value : values) {
            JsonObject model = value.getAsJsonObject();
            String name = requiredString(model, "model_name");
            models.add(new SupportedModel(name, model));
        }

        return new ServerCapabilities(List.copyOf(models), copy);
    }

    public static CheckpointPath checkpointPath(JsonObject source, String runId, String checkpointId) {
        JsonObject copy = source.deepCopy();
        String type = checkpointId.contains("/") ? checkpointId.substring(0, checkpointId.indexOf('/')) : "";
        String path = requiredString(copy, "path");
        String twinklePath = requiredString(copy, "twinkle_path");

        return new CheckpointPath(path, twinklePath, runId, type, checkpointId, copy);
    }

    public static DeleteCheckpointResult deleteCheckpoint(JsonObject source) {
        JsonObject copy = source.deepCopy();
        boolean success = copy.has("success") && copy.remove("success").getAsBoolean();
        String message = copy.has("message") ? copy.remove("message").getAsString() : "";

        return new DeleteCheckpointResult(success, message, copy);
    }

    public static WeightsInfo weightsInfo(JsonObject source) {
        JsonObject copy = source.deepCopy();
        Integer rank = copy.has("lora_rank") ? copy.remove("lora_rank").getAsInt() : null;
        boolean lora = copy.has("is_lora") && copy.remove("is_lora").getAsBoolean();

        return new WeightsInfo(
                requiredString(copy, "training_run_id"),
                requiredString(copy, "base_model"),
                requiredString(copy, "model_owner"),
                lora,
                rank,
                copy);
    }

    private static int requiredInt(JsonObject value, String name) {
        return required(value, name).getAsInt();
    }

    private static String requiredString(JsonObject value, String name) {
        return required(value, name).getAsString();
    }

    private static JsonElement required(JsonObject value, String name) {
        JsonElement result = value.remove(name);
        if (result == null || result.isJsonNull()) {
            throw new TwinkleProtocolException("Server response is missing required field: " + name, null);
        }

        return result;
    }
}
