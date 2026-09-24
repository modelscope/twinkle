package io.github.modelscope.twinkle.transport;

import com.google.gson.JsonElement;
import java.util.Map;

/** Defines the minimal HTTP operations used by resource clients. */
public interface HttpTransport extends AutoCloseable {
    JsonElement get(String path, Map<String, ?> query);

    JsonElement post(String path, Map<String, ?> payload);

    JsonElement delete(String path);

    void setSessionId(String sessionId);

    String sessionId();

    @Override
    void close();
}
