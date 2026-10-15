package io.github.modelscope.twinkle.types;

/**
 * Identifies an artifact saved by a model or sampler.
 *
 * @param twinklePath the artifact location in Twinkle storage
 * @param checkpointDir the local checkpoint directory on the server
 */
public record SaveResponse(String twinklePath, String checkpointDir) {}
