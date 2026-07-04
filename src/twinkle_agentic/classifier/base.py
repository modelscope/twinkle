from abc import ABC, abstractmethod


class Classifier(ABC):

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path

    @abstractmethod
    def classify(self, text: str) -> str:
        pass