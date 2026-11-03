package io.github.modelscope.twinkle.types;

import com.google.gson.JsonElement;
import java.util.List;

/**
 * Represents one generated sequence.
 *
 * @param stopReason the reason generation stopped
 * @param tokens the generated token identifiers
 * @param decoded the decoded text returned by the server
 * @param raw the original sequence JSON returned by the server
 */
public record SampledSequence(String stopReason, List<Integer> tokens, String decoded, JsonElement raw) {}
