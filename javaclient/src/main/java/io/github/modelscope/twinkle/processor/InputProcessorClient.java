package io.github.modelscope.twinkle.processor;

import com.google.gson.JsonElement;
import io.github.modelscope.twinkle.transport.HttpTransport;
import io.github.modelscope.twinkle.transport.JsonValue;
import java.util.List;
import java.util.Map;

/** Provides operations for a remote input processor. */
public final class InputProcessorClient extends RemoteProcessor {

    InputProcessorClient(HttpTransport transport, String id) {
        super(transport, id);
    }

    /** Processes a batch of inputs with the remote input processor. */
    public JsonElement process(List<JsonElement> inputs, Map<String, ?> options) {
        java.util.LinkedHashMap<String, Object> data = new java.util.LinkedHashMap<>();
        data.put("inputs", inputs.stream().map(JsonValue::new).toList());
        if (options != null) {
            data.putAll(options);
        }
        return call("__call__", data);
    }
}
