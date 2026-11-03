package io.github.modelscope.twinkle.transport;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import io.github.modelscope.twinkle.config.ClientConfig;
import io.github.modelscope.twinkle.exception.TwinkleIterationExhaustedException;
import io.github.modelscope.twinkle.exception.TwinkleProtocolException;
import io.github.modelscope.twinkle.exception.TwinkleServiceException;
import io.github.modelscope.twinkle.exception.TwinkleTransportException;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;
import okhttp3.HttpUrl;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/** Implements synchronous HTTP transport with OkHttp. */
public final class OkHttpTransport implements HttpTransport {

    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");
    private static final int MAX_RESPONSE_BODY_BYTES = 10 * 1024 * 1024;
    private final ClientConfig config;
    private final OkHttpClient client;
    private final TwinkleJsonCodec codec = new TwinkleJsonCodec();
    private final String requestId = UUID.randomUUID().toString();
    private volatile String sessionId;

    public OkHttpTransport(ClientConfig config) {
        this.config = config;
        this.client = new OkHttpClient.Builder()
                .connectTimeout(config.connectTimeout())
                .readTimeout(config.requestTimeout())
                .writeTimeout(config.requestTimeout())
                .callTimeout(config.requestTimeout())
                .followRedirects(false)
                .followSslRedirects(false)
                .build();
    }

    @Override
    public JsonElement get(String path, Map<String, ?> query) {
        HttpUrl.Builder url = url(path).newBuilder();
        if (query != null) {
            query.forEach((key, value) -> {
                if (value != null) {
                    url.addQueryParameter(key, String.valueOf(value));
                }
            });
        }
        return execute(new Request.Builder().url(url.build()).get());
    }

    @Override
    public JsonElement post(String path, Map<String, ?> payload) {
        String body = codec.gson().toJson(codec.encode(payload));
        return execute(new Request.Builder().url(url(path)).post(RequestBody.create(body, JSON)));
    }

    @Override
    public JsonElement delete(String path) {
        return execute(new Request.Builder().url(url(path)).delete());
    }

    @Override
    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    @Override
    public String sessionId() {
        return sessionId;
    }

    @Override
    public void close() {
        client.dispatcher().executorService().shutdown();
        client.connectionPool().evictAll();
    }

    private HttpUrl url(String path) {
        String normalized = path.startsWith("/") ? path : "/" + path;
        HttpUrl parsed = HttpUrl.parse(config.apiBaseUrl() + normalized);
        if (parsed == null) {
            throw new IllegalArgumentException("Invalid request URL: " + path);
        }
        return parsed;
    }

    private JsonElement execute(Request.Builder request) {
        String token = config.apiKeySupplier().get();
        if (token == null || token.isBlank()) {
            throw new IllegalStateException("apiKey must not be blank");
        }
        String authorization = "Bearer " + token;
        request.header("Authorization", authorization)
                .header("Twinkle-Authorization", authorization)
                .header("x-request-id", requestId)
                .header("X-Ray-Serve-Request-Id", requestId)
                .header("serve_multiplexed_model_id", requestId)
                .header("Serve-Multiplexed-Model-Id", requestId);
        if (sessionId != null && !sessionId.isBlank()) {
            request.header("X-Twinkle-Session-Id", sessionId);
        }
        try (Response response = client.newCall(request.build()).execute()) {
            String text = readBody(response);
            URI endpoint = response.request().url().uri();
            if (!response.isSuccessful()) {
                String detail = detail(text);
                if (response.code() == 410) {
                    throw new TwinkleIterationExhaustedException(endpoint, requestId, detail);
                }
                throw new TwinkleServiceException(response.code(), endpoint, requestId, detail);
            }
            try {
                return text.isBlank() ? new JsonObject() : JsonParser.parseString(text);
            } catch (RuntimeException error) {
                throw new TwinkleProtocolException("Server response is not valid JSON: " + endpoint, error);
            }
        } catch (TwinkleServiceException | TwinkleProtocolException error) {
            throw error;
        } catch (IOException error) {
            throw new TwinkleTransportException(
                    "HTTP request failed: " + request.build().url(), error);
        }
    }

    private static String readBody(Response response) throws IOException {
        if (response.body() == null) {
            return "";
        }
        long contentLength = response.body().contentLength();
        if (contentLength > MAX_RESPONSE_BODY_BYTES) {
            throw new TwinkleProtocolException(
                    "Server response body exceeds " + MAX_RESPONSE_BODY_BYTES + " bytes", null);
        }
        byte[] body = response.body().byteStream().readNBytes(MAX_RESPONSE_BODY_BYTES + 1);
        if (body.length > MAX_RESPONSE_BODY_BYTES) {
            throw new TwinkleProtocolException(
                    "Server response body exceeds " + MAX_RESPONSE_BODY_BYTES + " bytes", null);
        }
        return new String(body, StandardCharsets.UTF_8);
    }

    private String detail(String text) {
        try {
            JsonElement json = JsonParser.parseString(text);
            if (json.isJsonObject() && json.getAsJsonObject().has("detail")) {
                return json.getAsJsonObject().get("detail").getAsString();
            }
        } catch (RuntimeException ignored) {
            /* Use the original body when the response is not JSON. */
        }
        return text.isBlank() ? "Server did not provide error details" : text;
    }
}
