package io.github.modelscope.twinkle.types;

/** Enumerates dataset types and their corresponding server class names. */
public enum DatasetKind {
    DATASET("Dataset"),
    LAZY_DATASET("LazyDataset"),
    ITERABLE_DATASET("IterableDataset"),
    PACKING_DATASET("PackingDataset"),
    ITERABLE_PACKING_DATASET("IterablePackingDataset");

    private final String serverClassName;

    DatasetKind(String serverClassName) {
        this.serverClassName = serverClassName;
    }

    public String serverClassName() {
        return serverClassName;
    }
}
