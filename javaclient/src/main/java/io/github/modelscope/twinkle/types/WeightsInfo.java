package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;

/**
 * Describes the training metadata associated with a weight file.
 *
 * @param trainingRunId the identifier of the owning training run
 * @param baseModel the associated base model name
 * @param modelOwner the owner of the base model
 * @param isLora whether the weight file contains a LoRA adapter
 * @param loraRank the LoRA rank, or {@code null} for non-LoRA weights
 * @param extensions additional server-defined fields
 */
public record WeightsInfo(
        String trainingRunId,
        String baseModel,
        String modelOwner,
        boolean isLora,
        Integer loraRank,
        JsonObject extensions) {}
