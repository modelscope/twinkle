package io.github.modelscope.twinkle.transport;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import java.util.Collection;
import java.util.Map;

/** Recursively encodes request payloads for the Twinkle protocol. */
public final class TwinkleJsonCodec {

    private final Gson gson = new GsonBuilder().serializeNulls().create();

    public Gson gson() {
        return gson;
    }

    public JsonElement encode(Object value) {
        if (value == null) {
            return JsonNull.INSTANCE;
        }
        if (value instanceof JsonValue raw) {
            return raw.value();
        }
        if (value instanceof JsonElement json) {
            return json;
        }
        if (value instanceof TwinkleSerializable serializable) {
            return new JsonPrimitive(gson.toJson(serializable.toTwinkleJson()));
        }
        if (value instanceof Map<?, ?> map) {
            JsonObject object = new JsonObject();
            map.forEach((key, item) -> object.add(String.valueOf(key), encode(item)));
            return object;
        }
        if (value instanceof Collection<?> collection) {
            JsonArray array = new JsonArray();
            collection.forEach(item -> array.add(encode(item)));
            return array;
        }
        if (value.getClass().isArray()) {
            return gson.toJsonTree(value);
        }
        return gson.toJsonTree(value);
    }
}
