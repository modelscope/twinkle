package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;

/**
 * Describes the LoRA capacity reported by the server.
 *
 * @param maxLoras the maximum number of LoRA adapters allowed by the server
 * @param usedLoras the number of LoRA adapters currently in use
 * @param freeLoras the number of LoRA adapters still available
 * @param extensions additional server-defined fields
 */
public record CapacityInfo(int maxLoras, int usedLoras, int freeLoras, JsonObject extensions) {}
