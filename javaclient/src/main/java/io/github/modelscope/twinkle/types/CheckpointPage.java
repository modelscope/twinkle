package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;
import java.util.List;

/**
 * Represents one page of checkpoints.
 *
 * @param checkpoints the checkpoints in the current page
 * @param cursor the pagination cursor
 * @param extensions additional server-defined fields
 */
public record CheckpointPage(List<Checkpoint> checkpoints, Cursor cursor, JsonObject extensions) {}
