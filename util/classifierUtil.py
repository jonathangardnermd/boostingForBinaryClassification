from typing import List
from util.featureUtil import FeatureVector


class WeakClassifier:
    def __init__(self, classifier: callable):
        self.classifier = classifier
        self.idxsMisclassified = set()

    def classify(self, feature_vector: FeatureVector):
        return 1 if self.classifier(feature_vector) else -1

    @property
    def num_wrong(self):
        return len(self.idxsMisclassified)


class StrongClassifier:
    def __init__(self):
        self.classifiers: List[WeakClassifier] = []
        self.alphas: List[int] = []

    def add_classifier(self, classifier: WeakClassifier, alpha: float):
        self.classifiers.append(classifier)
        self.alphas.append(alpha)

    def classify(self, feature_vector):
        return sum(
            [
                alpha * classifier.classify(feature_vector)
                for alpha, classifier in zip(self.alphas, self.classifiers)
            ]
        )
