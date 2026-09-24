package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;

/**
 * Confirms a checkpoint deletion request.
 *
 * @param success whether the checkpoint was deleted
 * @param message the message returned by the server
 * @param extensions additional server-defined fields
 */
public record DeleteCheckpointResult(boolean success, String message, JsonObject extensions) {}
