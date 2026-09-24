package io.github.modelscope.twinkle.types;

/**
 * Identifies a page within a server-side list.
 *
 * @param limit the maximum number of records requested for the page
 * @param offset the zero-based offset of the page
 * @param totalCount the total number of records in the list
 */
public record Cursor(int limit, int offset, int totalCount) {}
