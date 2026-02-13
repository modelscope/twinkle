# Lazy Loading Dataset

The difference between lazy loading datasets and `Dataset` is that its encode process occurs during `__getitem__`. When you call `encode`, the dataset will only mark that encoding needs to be performed when actually fetching data.
This type of dataset is generally used for multimodal scenarios to prevent memory explosion.

Lazy loading datasets also have the `@remote_class` decorator and can run in Ray workers.
