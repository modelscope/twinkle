# Streaming Fixed-Length Packing Dataset

`IterablePackingDataset` is the same as `PackingDataset`, both used for automatic concatenation and packing of datasets. The difference is that `IterablePackingDataset` is adapted for streaming reading in large datasets or multimodal scenarios.

This dataset also requires an additional call to `pack_dataset()` to enable the packing process.
```python
dataset.pack_dataset()
```

This dataset also has the `@remote_class` decorator and can run in Ray workers.
