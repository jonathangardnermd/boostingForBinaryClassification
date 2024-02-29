import random
from util.featureUtil import NumericalFeature, Feature, FeatureVector


class InputFeatureVector(FeatureVector):
    def __init__(self, x: float, y: float, label: int):
        super().__init__(x, y)
        self.label = label


class InputData:
    def __init__(self, num_samples: int = 100, *args):
        self.features: list[Feature] = [
            NumericalFeature("x"),
            NumericalFeature("y"),
        ]
        self.trainingData = []
        self.labels = []
        for _ in range(num_samples):
            x = random.uniform(
                -10, 10
            )  # Adjust the range according to your requirements
            y = random.uniform(
                -10, 10
            )  # Adjust the range according to your requirements
            label = random.choice([-1, 1])
            self.trainingData.append(InputFeatureVector(x, y, label))
            self.labels.append(label)
