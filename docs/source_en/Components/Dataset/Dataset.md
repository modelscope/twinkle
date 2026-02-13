# Basic Dataset Components

## DatasetMeta

Open-source community datasets can be defined by three fields:

- Dataset name: Represents the dataset ID, e.g., `swift/self-cognition`.
- Subset name: A dataset may contain multiple subsets, and each subset may have a different format.
- Subset split: Common splits include train/test, etc., used for training, validation, etc.

Using the Hugging Face community's datasets library, you can see an example of loading a dataset:

```python
from datasets import load_dataset
train_data = load_dataset("glue", "mrpc", split="train")
```

In Twinkle's dataset input, the `DatasetMeta` class is used to express the input data format. This class contains:

```python
@dataclass
class DatasetMeta:
    dataset_id: str
    subset_name: str = 'default'
    split: str = 'train'
    data_slice: Iterable = None
```

The first three fields correspond to the dataset name, subset name, and split respectively. The fourth field `data_slice` is the data range to be selected, for example:

```python
dataset_meta = DatasetMeta(..., data_slice=range(100))
```

When using this class, developers don't need to worry about `data_slice` going out of bounds. Twinkle will perform repeated sampling based on the dataset length.

> Note: data_slice has no effect on streaming datasets.

## Dataset

Twinkle's Dataset is a lightweight wrapper around the actual dataset, including operations such as downloading, loading, mixing, preprocessing, and encoding.

1. Loading datasets

```python
from twinkle.dataset import Dataset, DatasetMeta

dataset = Dataset(DatasetMeta(dataset_id='ms://swift/self-cognition', data_slice=range(1500)))
```
The `ms://` prefix of the dataset represents downloading from the ModelScope community. If replaced with `hf://`, it will download from the Hugging Face community. If there is no prefix, it defaults to downloading from the Hugging Face community. You can also pass a local path:

```python
from twinkle.dataset import Dataset, DatasetMeta

dataset = Dataset(DatasetMeta(dataset_id='my/custom/dataset.jsonl', data_slice=range(1500)))
```

2. Setting template

The Template component is responsible for converting string/image multimodal raw data into model input tokens. The dataset can set a Template to complete the `encode` process.

```python
dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
```

The set_template method supports passing `kwargs` (such as `max_length` in the example) to be used as constructor parameters for `Template`.

3. Adding datasets

```python
dataset.add_dataset(DatasetMeta(dataset_id='ms://xxx/xxx', data_slice=range(1000)))
```

`add_dataset` can add other datasets on top of existing datasets and subsequently call `mix_dataset` to mix them together.

4. Preprocessing data

The data preprocessing (ETL) process is an important workflow for data cleaning and standardization. For example:

```json
{
  "query": "some query here",
  "response": "some response with extra info",
}
```

In this raw data, the response may contain non-standard information. Before starting training, the response needs to be filtered and fixed, and replaced with Twinkle's standard format. So you can write a method to process the corresponding data:

```python
from twinkle.data_format import Trajectory, Message
from twinkle.dataset import DatasetMeta
def preprocess_row(row):
    query = row['query']
    response = row['response']
    if not query or not response:
        return None
    # Fix response
    response = _do_some_fix_on_response(response)
    return Trajectory(
        messages=[
            Message(role='user', content=query),
            Message(role='assistant', content=response)
        ]
    )

dataset.map(preprocess_row, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
```

> Tips:
> 1. Currently, the map interface of Dataset does not support `batched=True` mode
> 2. If a row has a problem, return None, and dataset.map will automatically filter empty rows
> 3. Different datasets may have different preprocessing methods, so an additional `dataset_meta` parameter needs to be passed. If the `add_dataset` method has not been called, i.e., there is only one dataset in the Dataset, this parameter can be omitted

Similarly, Dataset provides a filter method:
```python
def filter_row(row):
    if ...:
        return False
    else:
        return True

dataset.filter(filter_row, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
```

5. Mixing datasets

After adding multiple datasets to the Dataset, you need to use `mix_dataset` to mix them.

```python
dataset.mix_dataset()
```

6. Encoding dataset

Before inputting to the model, the dataset must go through tokenization and encoding to be converted into tokens. This process is usually completed by the `tokenizer` component. However, in current large model training processes, tokenizer is generally not used directly. This is because model training requires preparation of additional fields, and simply performing the tokenizer.encode process is not sufficient.
In Twinkle, encoding the dataset is completed by the Template component. We have already described how to set up Template above. Now you can directly encode:

```python
dataset.encode()
```

> 1. Dataset's `map`, `encode`, `filter`, and other methods all use the `map` method of `datasets`, so you can use the corresponding parameters in the kwargs of the corresponding methods
> 2. The `load_from_cache_file` parameter defaults to False, because when this parameter is set to True, it can cause headaches when the dataset changes but training still uses the cache. If your dataset is large and updated infrequently, you can directly set it to True
> 3. encode does not need to specify `DatasetMeta` because after preprocessing, all datasets have the same format

6. Getting data

Like ordinary datasets, Twinkle's `Dataset` can use data through indexing.

```python
trajectory = dataset[0]
length = len(dataset)
```

7. Remote execution support

The `Dataset` class is marked with the `@remote_class` decorator, so it can run in Ray:

```python
dataset = Dataset(..., remote_group='actor_group')
# The following methods will run on Ray workers
dataset.map(...)
```

The Ray execution of the Dataset component is in `first` mode, meaning only one worker process runs and loads.

> The overall dataset usage workflow is:
> 1. Construct the dataset, passing in the remote_group parameter if running in a Ray worker
> 2. Set template
> 3. Preprocess data
> 4. If multiple datasets are added, mix the data
> 5. Encode data
