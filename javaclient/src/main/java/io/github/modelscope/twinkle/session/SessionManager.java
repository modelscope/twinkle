package io.github.modelscope.twinkle.session;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import io.github.modelscope.twinkle.exception.TwinkleProtocolException;
import io.github.modelscope.twinkle.transport.HttpTransport;
import java.time.Duration;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Creates a server session and manages its background heartbeat. */
public final class SessionManager implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(SessionManager.class.getName());
    private final HttpTransport transport;
    private final String routePrefix;
    private final ScheduledExecutorService executor;
    private final String sessionId;
    private volatile Throwable lastHeartbeatFailure;

    public SessionManager(
            HttpTransport transport,
            String routePrefix,
            Duration interval,
            Map<String, ?> metadata,
            String existingSessionId) {
        this.transport = transport;
        this.routePrefix = routePrefix;
        this.sessionId =
                existingSessionId == null || existingSessionId.isBlank() ? create(metadata) : existingSessionId;
        transport.setSessionId(sessionId);
        this.executor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "TwinkleSessionHeartbeat");
            thread.setDaemon(true);
            return thread;
        });
        long delay = interval.toMillis();
        executor.scheduleWithFixedDelay(this::heartbeat, delay, delay, TimeUnit.MILLISECONDS);
    }

    public String sessionId() {
        return sessionId;
    }

    /** Returns the most recent heartbeat failure for session health monitoring. */
    public Optional<Throwable> lastHeartbeatFailure() {
        return Optional.ofNullable(lastHeartbeatFailure);
    }

    private String create(Map<String, ?> metadata) {
        try {
            JsonElement response = transport.post(routePrefix + "/create_session", Map.of("metadata", metadata));
            JsonObject body = response.getAsJsonObject();
            JsonElement value = body.get("session_id");
            if (value == null || value.isJsonNull() || value.getAsString().isBlank()) {
                throw new TwinkleProtocolException("Create-session response is missing a valid session_id", null);
            }
            return value.getAsString();
        } catch (TwinkleProtocolException error) {
            throw error;
        } catch (RuntimeException error) {
            throw new TwinkleProtocolException("Invalid create-session response", error);
        }
    }

    private void heartbeat() {
        try {
            transport.post(routePrefix + "/session_heartbeat", Map.of("session_id", sessionId));
            lastHeartbeatFailure = null;
        } catch (RuntimeException error) {
            lastHeartbeatFailure = error;
            LOG.log(Level.WARNING, "Twinkle session heartbeat failed", error);
        }
    }

    @Override
    public void close() {
        executor.shutdownNow();
    }
}
