package io.github.modelscope.twinkle.model;

import io.github.modelscope.twinkle.transport.HttpTransport;

/** Creates remote training models. */
public final class ModelsClient {

    private final HttpTransport transport;

    public ModelsClient(HttpTransport transport, String ignoredRoutePrefix) {
        this.transport = transport;
    }

    /** Opens a remote training model on the Twinkle server. */
    public ModelClient open(String modelId) {
        return new ModelClient(transport, modelId);
    }
}
