package io.github.modelscope.twinkle.transport;

import com.google.gson.JsonObject;

/** Marks a value object that uses Twinkle's specialized JSON string protocol. */
public interface TwinkleSerializable {
    JsonObject toTwinkleJson();
}
