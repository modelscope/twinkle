package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;
import java.util.List;

/**
 * Describes the capabilities advertised by the server.
 *
 * @param supportedModels the base models currently supported by the server
 * @param extensions additional server-defined fields
 */
public record ServerCapabilities(List<SupportedModel> supportedModels, JsonObject extensions) {}
