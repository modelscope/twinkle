package io.github.modelscope.twinkle.types;

import com.google.gson.JsonObject;
import java.util.List;

/**
 * Represents one page of training runs.
 *
 * @param runs the training runs in the current page
 * @param cursor the pagination cursor
 * @param extensions additional server-defined fields
 */
public record TrainingRunPage(List<TrainingRun> runs, Cursor cursor, JsonObject extensions) {}
