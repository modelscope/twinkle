package io.github.modelscope.twinkle.types;

import java.util.List;

/**
 * Groups the generated candidates for one input.
 *
 * @param sequences the generated candidate sequences
 */
public record SampleResult(List<SampledSequence> sequences) {}
