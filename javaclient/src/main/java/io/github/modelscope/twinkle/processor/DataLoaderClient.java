package io.github.modelscope.twinkle.processor;

import com.google.gson.JsonElement;
import io.github.modelscope.twinkle.exception.TwinkleIterationExhaustedException;
import io.github.modelscope.twinkle.transport.HttpTransport;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;

/** Provides a remote data loader that supports Java for-each iteration. */
public final class DataLoaderClient extends RemoteProcessor implements Iterable<JsonElement> {

    DataLoaderClient(HttpTransport transport, String id) {
        super(transport, id);
    }

    /** Retrieves the number of batches in the remote data loader. */
    public int length() {
        return call("__len__", Map.of()).getAsInt();
    }

    /** Sets the processor used by the remote data loader. */
    public JsonElement setProcessor(String processorClass, Map<String, ?> options) {
        return call(
                "set_processor",
                options == null ? Map.of("processor_cls", processorClass) : merge(processorClass, options));
    }

    /** Skips consumed samples in the remote data loader. */
    public JsonElement skipConsumedSamples(int count) {
        return call("skip_consumed_samples", Map.of("consumed_train_samples", count));
    }

    /** Retrieves the current remote data loader state. */
    public JsonElement state() {
        return call("get_state", Map.of());
    }

    @Override
    /** Creates an iterator that reads remote data by batch. */
    public Iterator<JsonElement> iterator() {
        call("__iter__", Map.of());
        return new Iterator<>() {
            private boolean exhausted;
            private JsonElement nextValue;

            @Override
            public boolean hasNext() {
                if (exhausted) {
                    return false;
                }
                if (nextValue != null) {
                    return true;
                }
                try {
                    nextValue = call("__next__", Map.of());
                    return true;
                } catch (TwinkleIterationExhaustedException error) {
                    exhausted = true;
                    return false;
                }
            }

            @Override
            public JsonElement next() {
                if (!hasNext()) {
                    throw new NoSuchElementException("Remote data loader is exhausted");
                }
                JsonElement result = nextValue;
                nextValue = null;
                return result;
            }
        };
    }

    private static Map<String, Object> merge(String processorClass, Map<String, ?> options) {
        java.util.LinkedHashMap<String, Object> result = new java.util.LinkedHashMap<>();
        result.put("processor_cls", processorClass);
        result.putAll(options);
        return result;
    }
}
