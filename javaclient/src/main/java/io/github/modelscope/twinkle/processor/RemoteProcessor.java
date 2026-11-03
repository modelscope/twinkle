package io.github.modelscope.twinkle.processor;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.transport.HttpTransport;
import java.util.LinkedHashMap;
import java.util.Map;

/** Encapsulates shared invocation logic for remote processors. */
abstract class RemoteProcessor {

    protected final HttpTransport transport;
    protected final String processorId;

    RemoteProcessor(HttpTransport transport, String processorId) {
        this.transport = transport;
        this.processorId = processorId;
    }

    public String processorId() {
        return processorId;
    }

    protected JsonElement call(String function, Map<String, ?> arguments) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("processor_id", processorId);
        payload.put("function", function);
        if (arguments != null) {
            payload.putAll(arguments);
        }
        JsonObject response = transport.post("/processor/twinkle/call", payload).getAsJsonObject();
        return response.has("result") ? response.get("result") : response;
    }
}
