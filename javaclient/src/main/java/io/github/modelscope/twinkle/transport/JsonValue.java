package io.github.modelscope.twinkle.transport;

import com.google.gson.JsonElement;
import java.util.Objects;

/**
 * Wraps raw JSON to prevent it from being encoded as a string again.
 *
 * <p>For example, wrapping a batch such as {@code [{"input_ids":[1,2,3]}]} keeps the
 * {@code inputs} request field as a JSON array instead of producing the string
 * {@code "[{\"input_ids\":[1,2,3}]}"}.
 *
 * @param value the JSON element to embed without further encoding
 */
public record JsonValue(JsonElement value) {
    public JsonValue {
        Objects.requireNonNull(value, "value must not be null");
    }
}
