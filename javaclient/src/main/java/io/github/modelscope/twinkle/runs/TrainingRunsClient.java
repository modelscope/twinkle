package io.github.modelscope.twinkle.runs;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.internal.ResponseMapper;
import io.github.modelscope.twinkle.transport.HttpTransport;
import io.github.modelscope.twinkle.types.Checkpoint;
import io.github.modelscope.twinkle.types.CheckpointPath;
import io.github.modelscope.twinkle.types.Cursor;
import io.github.modelscope.twinkle.types.DeleteCheckpointResult;
import io.github.modelscope.twinkle.types.TrainingRun;
import io.github.modelscope.twinkle.types.WeightsInfo;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Provides queries for training runs and checkpoints. */
public final class TrainingRunsClient {

    private final HttpTransport transport;
    private final String prefix;

    public TrainingRunsClient(HttpTransport transport, String prefix) {
        this.transport = transport;
        this.prefix = prefix;
    }

    /** Lists training runs on the Twinkle server. */
    public Page list(int limit, int offset, boolean allUsers) {
        if (limit <= 0 || offset < 0) {
            throw new IllegalArgumentException("limit must be greater than 0 and offset must not be negative");
        }
        JsonObject body = transport
                .get(prefix + "/training_runs", Map.of("limit", limit, "offset", offset, "all_users", allUsers))
                .getAsJsonObject();
        List<TrainingRun> runs = new ArrayList<>();
        for (JsonElement item : body.getAsJsonArray("training_runs")) {
            runs.add(toRun(item.getAsJsonObject()));
        }

        JsonObject cursor = body.has("cursor") ? body.getAsJsonObject("cursor") : new JsonObject();
        return new Page(
                List.copyOf(runs),
                new Cursor(intValue(cursor, "limit"), intValue(cursor, "offset"), intValue(cursor, "total_count")));
    }

    /** Retrieves a training run by identifier. */
    public TrainingRun get(String runId) {
        return toRun(transport
                .get(prefix + "/training_runs/" + requireId(runId), Map.of())
                .getAsJsonObject());
    }

    /** Lists checkpoints for a training run. */
    public List<Checkpoint> listCheckpoints(String runId) {
        JsonArray values = transport
                .get(prefix + "/training_runs/" + requireId(runId) + "/checkpoints", Map.of())
                .getAsJsonObject()
                .getAsJsonArray("checkpoints");
        List<Checkpoint> result = new ArrayList<>();
        for (JsonElement item : values) {
            result.add(toCheckpoint(item.getAsJsonObject()));
        }
        return List.copyOf(result);
    }

    /** Retrieves the storage locations of a checkpoint. */
    public CheckpointPath checkpointPath(String runId, String checkpointId) {
        return ResponseMapper.checkpointPath(
                transport
                        .get(prefix + "/checkpoint_path/" + requireId(runId) + "/" + requireId(checkpointId), Map.of())
                        .getAsJsonObject(),
                runId,
                checkpointId);
    }

    /** Deletes a checkpoint from a training run. */
    public DeleteCheckpointResult deleteCheckpoint(String runId, String checkpointId) {
        return ResponseMapper.deleteCheckpoint(transport
                .delete(prefix + "/training_runs/" + requireId(runId) + "/checkpoints/" + requireId(checkpointId))
                .getAsJsonObject());
    }

    /** Retrieves training metadata for a weight file. */
    public WeightsInfo weightsInfo(String twinklePath) {
        return ResponseMapper.weightsInfo(transport
                .post(prefix + "/weights_info", Map.of("twinkle_path", requireId(twinklePath)))
                .getAsJsonObject());
    }

    public String latestCheckpointPath(String runId) {
        List<Checkpoint> checkpoints = listCheckpoints(runId);
        return checkpoints.isEmpty()
                ? null
                : checkpointPath(runId, checkpoints.get(checkpoints.size() - 1).checkpointId())
                        .path();
    }

    private static TrainingRun toRun(JsonObject value) {
        return new TrainingRun(
                string(value, "training_run_id"), string(value, "base_model"), string(value, "model_owner"), value);
    }

    private static Checkpoint toCheckpoint(JsonObject value) {
        return new Checkpoint(
                string(value, "checkpoint_id"), string(value, "checkpoint_type"), string(value, "twinkle_path"), value);
    }

    private static String string(JsonObject value, String key) {
        return value.has(key) && !value.get(key).isJsonNull() ? value.get(key).getAsString() : null;
    }

    private static int intValue(JsonObject value, String key) {
        return value.has(key) ? value.get(key).getAsInt() : 0;
    }

    private static String requireId(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("identifier must not be blank");
        }
        return URLEncoder.encode(value, StandardCharsets.UTF_8).replace("+", "%20");
    }

    /**
     * Represents a page of training runs.
     *
     * @param runs the training runs in the current page
     * @param cursor the pagination cursor
     */
    public record Page(List<TrainingRun> runs, Cursor cursor) {}
}
