package io.github.modelscope.twinkle;

import io.github.modelscope.twinkle.config.ClientConfig;
import io.github.modelscope.twinkle.internal.ResponseMapper;
import io.github.modelscope.twinkle.model.ModelsClient;
import io.github.modelscope.twinkle.processor.ProcessorsClient;
import io.github.modelscope.twinkle.runs.TrainingRunsClient;
import io.github.modelscope.twinkle.sampler.SamplersClient;
import io.github.modelscope.twinkle.session.SessionManager;
import io.github.modelscope.twinkle.transport.HttpTransport;
import io.github.modelscope.twinkle.transport.OkHttpTransport;
import io.github.modelscope.twinkle.types.CapacityInfo;
import io.github.modelscope.twinkle.types.ServerCapabilities;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;

/** Provides the primary entry point for the Twinkle Java SDK. */
public final class TwinkleClient implements AutoCloseable {

    private final HttpTransport transport;
    private final String configRoutePrefix;
    private final SessionManager session;
    private final TrainingRunsClient trainingRuns;
    private final ModelsClient models;
    private final SamplersClient samplers;
    private final ProcessorsClient processors;

    private TwinkleClient(Builder builder) {
        ClientConfig config = builder.config.build();
        this.transport = new OkHttpTransport(config);
        this.configRoutePrefix = config.routePrefix();
        this.session = new SessionManager(
                transport,
                config.routePrefix(),
                config.heartbeatInterval(),
                builder.metadata,
                builder.existingSessionId);
        this.trainingRuns = new TrainingRunsClient(transport, config.routePrefix());
        this.models = new ModelsClient(transport, config.routePrefix());
        this.samplers = new SamplersClient(transport, config.routePrefix());
        this.processors = new ProcessorsClient(transport, config.routePrefix());
    }

    public static Builder builder() {
        return new Builder();
    }

    public String sessionId() {
        return session.sessionId();
    }

    /** Returns the most recent background heartbeat failure, if any. */
    public Optional<Throwable> lastHeartbeatFailure() {
        return session.lastHeartbeatFailure();
    }

    public TrainingRunsClient trainingRuns() {
        return trainingRuns;
    }

    public ModelsClient models() {
        return models;
    }

    public SamplersClient samplers() {
        return samplers;
    }

    public ProcessorsClient processors() {
        return processors;
    }

    /** Checks whether the Twinkle server is reachable. */
    public boolean healthCheck() {
        try {
            transport.get(configuredRoute("/healthz"), Map.of());
            return true;
        } catch (RuntimeException error) {
            return false;
        }
    }

    /** Retrieves the base models supported by the Twinkle server. */
    public ServerCapabilities serverCapabilities() {
        return ResponseMapper.serverCapabilities(transport
                .get(configuredRoute("/get_server_capabilities"), Map.of())
                .getAsJsonObject());
    }

    /** Retrieves the LoRA capacity reported by the Twinkle server. */
    public CapacityInfo capacityInfo() {
        return ResponseMapper.capacityInfo(
                transport.get(configuredRoute("/capacity_info"), Map.of()).getAsJsonObject());
    }

    @Override
    public void close() {
        session.close();
        transport.close();
    }

    private String configuredRoute(String endpoint) {
        return configRoutePrefix + endpoint;
    }

    /** Builds a client with server, authentication, and session settings. */
    public static final class Builder {

        private final ClientConfig.Builder config = ClientConfig.builder();
        private Map<String, ?> metadata = Map.of();
        private String existingSessionId;

        public Builder baseUrl(String value) {
            config.baseUrl(value);
            return this;
        }

        public Builder apiKey(String value) {
            config.apiKey(value);
            return this;
        }

        public Builder routePrefix(String value) {
            config.routePrefix(value);
            return this;
        }

        public Builder connectTimeout(Duration value) {
            config.connectTimeout(value);
            return this;
        }

        public Builder requestTimeout(Duration value) {
            config.requestTimeout(value);
            return this;
        }

        public Builder heartbeatInterval(Duration value) {
            config.heartbeatInterval(value);
            return this;
        }

        public Builder sessionMetadata(Map<String, ?> value) {
            metadata = value == null ? Map.of() : Map.copyOf(value);
            return this;
        }

        public Builder existingSessionId(String value) {
            existingSessionId = value;
            return this;
        }

        public TwinkleClient build() {
            return new TwinkleClient(this);
        }
    }
}
