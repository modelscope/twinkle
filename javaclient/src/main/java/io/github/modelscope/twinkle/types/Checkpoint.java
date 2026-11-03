package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;

/**
 * Summarizes a training checkpoint.
 *
 * @param checkpointId the unique checkpoint identifier
 * @param checkpointType the checkpoint type
 * @param twinklePath the checkpoint location in Twinkle storage
 * @param raw unmodeled fields returned by the server
 */
public record Checkpoint(String checkpointId, String checkpointType, String twinklePath, JsonObject raw) {}
