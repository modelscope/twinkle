package io.github.modelscope.twinkle.types;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.transport.TwinkleSerializable;
import java.util.List;

/**
 * Locates a dataset or supplies inline dataset content.
 *
 * @param datasetId the dataset identifier or storage location
 * @param subsetName the dataset subset name
 * @param split the dataset split, such as {@code train}
 * @param dataSlice the range or index slice to read
 * @param data inline dataset content sent to the server
 */
public record DatasetMeta(String datasetId, String subsetName, String split, Object dataSlice, Object data)
        implements TwinkleSerializable {
    public DatasetMeta {
        if ((datasetId == null || datasetId.isBlank()) && data == null) {
            throw new IllegalArgumentException("datasetId and data must not both be empty");
        }
    }

    public static DatasetMeta of(String datasetId) {
        return new DatasetMeta(datasetId, "default", "train", null, null);
    }

    /** Creates a data slice equivalent to Python {@code range}. */
    public static JsonObject range(int start, int stop, int step) {
        if (step == 0) {
            throw new IllegalArgumentException("step must not be 0");
        }
        JsonObject value = new JsonObject();
        value.addProperty("_slice_type_", "range");
        value.addProperty("start", start);
        value.addProperty("stop", stop);
        value.addProperty("step", step);
        return value;
    }

    /** Creates a data slice from a list of indices. */
    public static JsonObject indices(List<Integer> values) {
        JsonObject value = new JsonObject();
        value.addProperty("_slice_type_", "list");
        JsonArray array = new JsonArray();
        values.forEach(array::add);
        value.add("values", array);
        return value;
    }

    @Override
    public JsonObject toTwinkleJson() {
        JsonObject value = new JsonObject();
        value.addProperty("_TWINKLE_TYPE_", "DatasetMeta");
        if (datasetId != null) {
            value.addProperty("dataset_id", datasetId);
        }
        if (subsetName != null) {
            value.addProperty("subset_name", subsetName);
        }
        if (split != null) {
            value.addProperty("split", split);
        }
        if (dataSlice != null) {
            value.add("data_slice", new Gson().toJsonTree(dataSlice));
        }
        if (data != null) {
            value.add("data", new Gson().toJsonTree(data));
        }
        return value;
    }
}
