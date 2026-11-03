package io.github.modelscope.twinkle.types;

import com.google.gson.JsonElement;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Defines a request to a remote sampler.
 *
 * @param inputs the JSON inputs to sample
 * @param samplingParams server sampling options, such as temperature
 * @param adapterName the LoRA adapter to use for this request
 * @param adapterUri an optional location from which the adapter can be loaded
 * @param numSamples the number of candidates to generate for each input
 */
public record SampleRequest(
        List<JsonElement> inputs,
        Map<String, ?> samplingParams,
        String adapterName,
        String adapterUri,
        int numSamples) {
    public SampleRequest {
        Objects.requireNonNull(inputs, "inputs must not be null");
        if (numSamples <= 0) throw new IllegalArgumentException("numSamples must be greater than 0");
        adapterName = adapterName == null ? "" : adapterName;
    }
}
