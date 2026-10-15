package io.github.modelscope.twinkle.processor;

import com.google.gson.JsonElement;
import io.github.modelscope.twinkle.transport.HttpTransport;
import io.github.modelscope.twinkle.types.DatasetKind;
import io.github.modelscope.twinkle.types.DatasetMeta;
import java.util.LinkedHashMap;
import java.util.Map;

/** Provides operations for a remote dataset. */
public final class DatasetClient extends RemoteProcessor {

    private final DatasetKind kind;

    DatasetClient(HttpTransport transport, String id, DatasetKind kind) {
        super(transport, id);
        this.kind = kind;
    }

    public DatasetKind kind() {
        return kind;
    }

    /** Sets the data template used by the remote dataset. */
    public JsonElement setTemplate(String templateFunction, Map<String, ?> options) {
        return call("set_template", merge(values("template_func", templateFunction), options));
    }

    /** Encodes the remote dataset into model inputs. */
    public JsonElement encode(boolean addGenerationPrompt, Map<String, ?> options) {
        return call("encode", merge(Map.of("add_generation_prompt", addGenerationPrompt), options));
    }

    /** Validates the remote dataset content. */
    public JsonElement check(Map<String, ?> options) {
        return call("check", options);
    }

    /** Casts the type of a remote dataset column. */
    public JsonElement castColumn(String column, boolean decode) {
        return call("cast_column", Map.of("column", column, "decode", decode));
    }

    /** Maps a function over the remote dataset. */
    public JsonElement map(String function, DatasetMeta meta, Map<String, ?> initArgs, Map<String, ?> options) {
        return call(
                "map",
                merge(values("preprocess_func", function, "dataset_meta", meta, "init_args", initArgs), options));
    }

    /** Filters the remote dataset. */
    public JsonElement filter(String function, DatasetMeta meta, Map<String, ?> initArgs, Map<String, ?> options) {
        return call(
                "filter", merge(values("filter_func", function, "dataset_meta", meta, "init_args", initArgs), options));
    }

    /** Adds another dataset to the remote dataset. */
    public JsonElement addDataset(DatasetMeta meta, Map<String, ?> options) {
        return call("add_dataset", merge(Map.of("dataset_meta", meta), options));
    }

    /** Mixes datasets held by the remote dataset. */
    public JsonElement mixDataset(boolean interleave) {
        return call("mix_dataset", Map.of("interleave", interleave));
    }

    /** Saves the remote dataset in the requested format. */
    public JsonElement saveAs(String outputPath, String format, int batchSize, String mode, Map<String, ?> options) {
        return call(
                "save_as",
                merge(
                        values("output_path", outputPath, "format", format, "batch_size", batchSize, "mode", mode),
                        options));
    }

    /** Flushes pending remote dataset output. */
    public JsonElement flushSave() {
        return call("flush_save", Map.of());
    }

    /** Retrieves an item from the remote dataset by index. */
    public JsonElement getItem(int index) {
        return call("__getitem__", Map.of("idx", index));
    }

    /** Retrieves the number of samples in the remote dataset. */
    public int length() {
        return call("__len__", Map.of()).getAsInt();
    }

    /** Packs the remote dataset on the server. */
    public JsonElement packDataset() {
        return call("pack_dataset", Map.of());
    }

    private static Map<String, Object> merge(Map<String, ?> first, Map<String, ?> second) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.putAll(first);
        if (second != null) {
            result.putAll(second);
        }
        return result;
    }

    private static Map<String, Object> values(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int index = 0; index < pairs.length; index += 2) {
            result.put((String) pairs[index], pairs[index + 1]);
        }
        return result;
    }
}
