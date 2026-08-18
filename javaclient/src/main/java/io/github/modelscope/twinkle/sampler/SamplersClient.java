package io.github.modelscope.twinkle.sampler;

import io.github.modelscope.twinkle.transport.HttpTransport;

/** Creates remote samplers. */
public final class SamplersClient {

    private final HttpTransport transport;

    public SamplersClient(HttpTransport transport, String ignoredRoutePrefix) {
        this.transport = transport;
    }

    /** Creates a remote sampler on the Twinkle server. */
    public SamplerClient open(String modelId) {
        return new SamplerClient(transport, modelId);
    }
}
