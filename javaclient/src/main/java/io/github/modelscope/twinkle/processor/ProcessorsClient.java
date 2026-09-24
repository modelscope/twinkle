package io.github.modelscope.twinkle.processor;

import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.transport.HttpTransport;
import io.github.modelscope.twinkle.types.DatasetKind;
import java.util.LinkedHashMap;
import java.util.Map;

/** Creates remote datasets, data loaders, and input processors. */
public final class ProcessorsClient {

    private final HttpTransport transport;

    public ProcessorsClient(HttpTransport transport, String ignoredRoutePrefix) {
        this.transport = transport;
    }

    /** Creates a remote dataset on the Twinkle server. */
    public DatasetClient dataset(DatasetKind kind, Map<String, ?> options) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("processor_type", "dataset");
        data.put("class_type", kind.serverClassName());
        if (options != null) {
            data.putAll(options);
        }
        return new DatasetClient(transport, create(data), kind);
    }

    /** Creates a data loader for a remote dataset. */
    public DataLoaderClient dataLoader(String datasetProcessorId, Map<String, ?> options) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("processor_type", "dataloader");
        data.put("class_type", "DataLoader");
        data.put("dataset", datasetProcessorId);
        if (options != null) {
            data.putAll(options);
        }
        return new DataLoaderClient(transport, create(data));
    }

    /** Creates an input processor on the Twinkle server. */
    public InputProcessorClient inputProcessor(Map<String, ?> options) {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("processor_type", "processor");
        data.put("class_type", "InputProcessor");
        if (options != null) {
            data.putAll(options);
        }
        return new InputProcessorClient(transport, create(data));
    }

    private String create(Map<String, ?> data) {
        JsonObject response = transport.post("/processor/twinkle/create", data).getAsJsonObject();
        return response.get("processor_id").getAsString();
    }
}
