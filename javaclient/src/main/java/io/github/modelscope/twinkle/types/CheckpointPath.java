package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;

/**
 * Identifies the local and Twinkle locations of a checkpoint.
 *
 * @param path the local path in the server environment
 * @param twinklePath the checkpoint location in Twinkle storage
 * @param trainingRunId the owning training run identifier
 * @param checkpointType the checkpoint type
 * @param checkpointId the checkpoint identifier
 * @param extensions additional server-defined fields
 */
public record CheckpointPath(
        String path,
        String twinklePath,
        String trainingRunId,
        String checkpointType,
        String checkpointId,
        JsonObject extensions) {}
