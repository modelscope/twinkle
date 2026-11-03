package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;

/**
 * Summarizes a server-side training run.
 *
 * @param trainingRunId the unique training run identifier
 * @param baseModel the base model used for training
 * @param modelOwner the owner of the base model
 * @param raw unmodeled fields returned by the server
 */
public record TrainingRun(String trainingRunId, String baseModel, String modelOwner, JsonObject raw) {}
