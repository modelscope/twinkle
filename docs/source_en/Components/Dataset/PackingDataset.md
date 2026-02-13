# Fixed-Length Packing Dataset

Packing datasets are used to concatenate variable-length data to a specified length. For example:

The dataset contains 4 pieces of data with length 5, and the Template component's max_length can accept a length of 10. The packing dataset will pre-fetch the data and concatenate it into 2 samples with length 10.

```text
ABCDE
FGHIJ
KLMNO
PQRST
```

Will be converted to
```text
ABCDEFGHIJ
KLMNOPQRST
```
Note that this concatenation occurs after `encode`, i.e., on the actual model input length. In the process, the dataset will perform the following operations:

1. Fetch `buffer length` samples
2. Encode these samples
3. Calculate based on the length of each sample using an automatic packing algorithm to find an optimal solution that minimizes the number of batches and makes the length of each sample closest to `max_length`
4. Add a `position_ids` field to distinguish different samples.

The final data format is similar to:

```json
{
  "input_ids": [1,2,3,4,5,6,7,8,9,10],
  "position_ids": [0,1,2,3,4,0,1,2,3,4],
  ...
}
```

The use of the dataset has the following differences from `Dataset`:

1. Must set `Template`
2. After calling `encode`, you need to call the `pack_dataset` method for final packing

```python
dataset.pack_dataset()
```

This dataset also has the `@remote_class` decorator and can run in Ray workers.
