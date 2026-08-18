package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;

/**
 * Describes a base model supported by the server.
 *
 * @param modelName the supported model name
 * @param extensions additional server-defined model fields
 */
public record SupportedModel(String modelName, JsonObject extensions) {}
