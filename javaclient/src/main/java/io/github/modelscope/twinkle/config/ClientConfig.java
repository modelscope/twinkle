package io.github.modelscope.twinkle.config;

import java.time.Duration;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Holds immutable client configuration.
 *
 * @param apiBaseUrl the normalized server API base URL
 * @param routePrefix the management API route prefix
 * @param apiKeySupplier the supplier for the authentication token
 * @param connectTimeout the connection timeout
 * @param requestTimeout the request timeout
 * @param heartbeatInterval the session heartbeat interval
 */
public record ClientConfig(
        String apiBaseUrl,
        String routePrefix,
        Supplier<String> apiKeySupplier,
        Duration connectTimeout,
        Duration requestTimeout,
        Duration heartbeatInterval) {
    public static Builder builder() {
        return new Builder();
    }

    /** Collects client options and validates them before construction. */
    public static final class Builder {

        private String baseUrl;
        private String routePrefix = "/twinkle";
        private Supplier<String> apiKeySupplier;
        private Duration connectTimeout = Duration.ofSeconds(30);
        private Duration requestTimeout = Duration.ofMinutes(10);
        private Duration heartbeatInterval = Duration.ofSeconds(10);

        public Builder baseUrl(String value) {
            this.baseUrl = value;
            return this;
        }

        public Builder routePrefix(String value) {
            this.routePrefix = value;
            return this;
        }

        public Builder apiKey(String value) {
            this.apiKeySupplier = () -> value;
            return this;
        }

        public Builder apiKeySupplier(Supplier<String> value) {
            this.apiKeySupplier = value;
            return this;
        }

        public Builder connectTimeout(Duration value) {
            this.connectTimeout = value;
            return this;
        }

        public Builder requestTimeout(Duration value) {
            this.requestTimeout = value;
            return this;
        }

        public Builder heartbeatInterval(Duration value) {
            this.heartbeatInterval = value;
            return this;
        }

        public ClientConfig build() {
            String server = normalizeBaseUrl(baseUrl);
            String prefix = normalizePrefix(routePrefix);
            Supplier<String> supplier = apiKeySupplier == null ? () -> "EMPTY_TOKEN" : apiKeySupplier;
            validateDuration(connectTimeout, "connectTimeout");
            validateDuration(requestTimeout, "requestTimeout");
            validateDuration(heartbeatInterval, "heartbeatInterval");
            if (heartbeatInterval.toMillis() == 0) {
                throw new IllegalArgumentException("heartbeatInterval must be at least 1 millisecond");
            }
            String token = Objects.requireNonNull(supplier.get(), "apiKey must not be null")
                    .trim();
            if (token.isEmpty()) {
                throw new IllegalArgumentException("apiKey must not be blank");
            }
            return new ClientConfig(server, prefix, supplier, connectTimeout, requestTimeout, heartbeatInterval);
        }

        /** Normalizes the server URL and ensures that it contains the API v1 path. */
        private static String normalizeBaseUrl(String value) {
            String result = value == null || value.isBlank() ? "http://127.0.0.1:8000" : value.trim();
            result = result.replaceAll("/+$", "");
            return result.endsWith("/api/v1") ? result : result + "/api/v1";
        }

        /** Normalizes the management API route prefix. */
        private static String normalizePrefix(String value) {
            if (value == null || value.isBlank()) {
                return "";
            }
            String result = value.trim().replaceAll("/+$", "");
            if (!result.startsWith("/")) {
                throw new IllegalArgumentException("routePrefix must start with /");
            }
            return result;
        }

        /** Validates that a timeout or heartbeat interval is positive. */
        private static void validateDuration(Duration value, String name) {
            if (value == null || value.isZero() || value.isNegative()) {
                throw new IllegalArgumentException(name + " must be greater than 0");
            }
        }
    }
}
