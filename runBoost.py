import math
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Iterator
from util.featureUtil import Feature, FeatureVector
from util.classifierUtil import WeakClassifier, StrongClassifier
from input.inputData2 import InputData

DEBUG_MODE = False
SHOW_PLOT = True
NUM_BOOSTING_ITERATIONS = 300
NUM_TRAINING_DATA_PTS = 100


def printDebug(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)


class ClassifierGenerator:
    def __init__(
        self,
        features: List[Feature],
        trainingData: List[FeatureVector],
        labels: List[int],
    ):
        self.features = features
        self.trainingData = trainingData
        self.labels = labels

    def find_all_numerical_classifiers(self, featIdx, labels, vals):
        z = zip(labels, vals)
        z = sorted(z, key=lambda x: x[1])
        for i in range(len(z) - 1):
            if z[i][1] != z[i + 1][1]:
                cutoff = (z[i][1] + z[i + 1][1]) / 2
                yield lambda x, cutoff=cutoff: x[featIdx] > cutoff
                yield lambda x, cutoff=cutoff: x[featIdx] <= cutoff  # mirror classifier

    def generate_classifiers(self) -> List[WeakClassifier]:
        classifiers = []
        for i in range(len(self.features)):
            if self.features[i].isBinary():
                classifiers.append(lambda x, i=i: bool(x[i]))
                classifiers.append(lambda x, i=i: not x[i])  # mirror classifier
            else:
                for classifier in self.find_all_numerical_classifiers(
                    i, self.labels, [x.data[i] for x in self.trainingData]
                ):
                    classifiers.append(classifier)
        classifiers.append(lambda x: True)
        classifiers.append(lambda x: False)  # mirror classifier
        return [WeakClassifier(classifier) for classifier in classifiers]


class BagOfClassifiers:
    @staticmethod
    def classifier_iter(
        classifiers,
    ) -> Iterator[Tuple[WeakClassifier, WeakClassifier]]:
        for classifierIdx in range(int(len(classifiers) / 2)):
            yield classifiers[2 * classifierIdx], classifiers[2 * classifierIdx + 1]

    @staticmethod
    def classify_all(
        classifiers: List[WeakClassifier],
        training_data: List[FeatureVector],
        labels: List[int],
    ):
        for classifier, mirrorClassifier in BagOfClassifiers.classifier_iter(
            classifiers
        ):
            for trainingPtIdx in range(len(training_data)):
                training_pt = training_data[trainingPtIdx]
                classification = classifier.classify(training_pt.data)
                if classification != labels[trainingPtIdx]:
                    classifier.idxsMisclassified.add(trainingPtIdx)
                else:
                    mirrorClassifier.idxsMisclassified.add(trainingPtIdx)

    @staticmethod
    def print_wrong_by_classifier(classifiers: List[WeakClassifier]):
        s = "\n".join(
            [
                f"classifierIdx={i}: {[idx+1 for idx in classifier.idxsMisclassified]}"
                for i, classifier in enumerate(classifiers)
            ]
        )
        printDebug(s)

    @staticmethod
    def eliminate_dominated_classifiers(classifiers: List[WeakClassifier]):
        classifier_idxs_to_toss = BagOfClassifiers.find_strict_subsets(classifiers)
        classifiers = [
            classifiers[i]
            for i in range(len(classifiers))
            if i not in classifier_idxs_to_toss
        ]
        return classifiers

    @staticmethod
    def find_strict_subsets(classifiers: List[WeakClassifier]):
        classifier_idxs_to_toss: Set[int] = set()
        for i in range(len(classifiers)):
            for j in range(i + 1, len(classifiers)):
                iWrongs = classifiers[i].idxsMisclassified
                jWrongs = classifiers[j].idxsMisclassified
                i_set_is_smaller = len(iWrongs) < len(jWrongs)

                if i_set_is_smaller and iWrongs.issubset(jWrongs):
                    classifier_idxs_to_toss.add(j)
                elif not i_set_is_smaller and jWrongs.issubset(iWrongs):
                    classifier_idxs_to_toss.add(i)
        for idx in classifier_idxs_to_toss:
            printDebug(
                f"tossing classifier {idx}: {classifiers[idx].idxsMisclassified}"
            )
        return classifier_idxs_to_toss


class Booster:
    @staticmethod
    def init_weights(N: int) -> list[float]:
        """Initialize weights for boosting."""
        return [1 / N for _ in range(N)]

    @staticmethod
    def alpha(error: float) -> float:
        """Calculate the weight of a weak classifier."""
        return 0.5 * math.log((1 - error) / error)

    @staticmethod
    def generate_strong_classifier_iter(
        num_boosting_iterations: int,
        num_training_data_pts: int,
        classifiers: list[WeakClassifier],
    ) -> Iterator[StrongClassifier]:
        """Generate a strong classifier using boosting."""
        weights = Booster.init_weights(num_training_data_pts)
        strong_classifier = StrongClassifier()
        for _ in range(num_boosting_iterations):
            min_error_idx, min_error = Booster.find_min_error(classifiers, weights)
            alpha_val = Booster.alpha(min_error)
            strong_classifier.add_classifier(classifiers[min_error_idx], alpha_val)
            weights = Booster.rescale_weights(
                weights, classifiers[min_error_idx].idxsMisclassified
            )
            yield strong_classifier

    @staticmethod
    def classify_training_data(
        strong_classifier: StrongClassifier,
        training_data: list[FeatureVector],
        labels: list[int],
    ) -> float:
        """Classify training data using the strong classifier."""
        numCorrect = 0
        for i in range(len(training_data)):
            classification_score = strong_classifier.classify(training_data[i].data)
            classification = 1 if classification_score > 0 else -1
            label = labels[i]
            if classification == label:
                numCorrect += 1
            printDebug(
                f"classificationScore={classification_score}, label={label}, correct={classification == label}"
            )
        return numCorrect / len(training_data)

    @staticmethod
    def rescale_weights(weights: list[float], wrongs: set[int]) -> list[float]:
        """Rescale weights based on misclassifications."""
        sum_wrongs = sum(weights[i] for i in wrongs)
        sum_rights = sum(weights[i] for i in range(len(weights)) if i not in wrongs)
        for i in range(len(weights)):
            if i in wrongs:
                weights[i] *= 0.5 / sum_wrongs
            else:
                weights[i] *= 0.5 / sum_rights
        return weights

    @staticmethod
    def find_min_error(
        classifiers: list[WeakClassifier], weights: list[float]
    ) -> Tuple[int, float]:
        """Find the weak classifier with minimum error."""
        min_error_idx = -1
        min_error = float("inf")
        for i, classifier in enumerate(classifiers):
            error = sum(weights[idx] for idx in classifier.idxsMisclassified)
            if error < min_error:
                min_error = error
                min_error_idx = i
        return min_error_idx, min_error


def plot():
    # make two plots:
    #   plot1: the distribution of the 2D training data with the two colors indicating the binary labels
    #   plot2: the accuracy as a function of the number of boosting iterations
    plt.figure(figsize=(14, 7))  # Adjust figure size for square plots
    plot_2D_training_data(training_data, labels)
    plot_accuracy(accuracies)
    plt.tight_layout()  # Adjust subplots to prevent overlap
    plt.show()


def plot_2D_training_data(training_data: List[FeatureVector], labels: List[int]):
    plt.subplot(1, 2, 1)  # Create subplot 1
    x = [x.data[0] for x in training_data]
    y = [x.data[1] for x in training_data]
    plt.scatter(x, y, c=labels)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("2D Training Data")


def plot_accuracy(accuracies: List[Tuple[int, float]]):
    plt.subplot(1, 2, 2)  # Create subplot 2
    plt.plot([x[0] for x in accuracies], [x[1] for x in accuracies])
    plt.xlabel("num boosting iterations")
    plt.ylabel("accuracy")
    plt.title("Accuracy")


##########################################################################
# Time to run the boosting algorithm
##########################################################################
input_data = InputData(NUM_TRAINING_DATA_PTS)
features = input_data.features
training_data = input_data.trainingData
labels = input_data.labels

# generate classifiers
classifiers = ClassifierGenerator(
    features, training_data, labels
).generate_classifiers()

# classify all training data with all classifiers
BagOfClassifiers.classify_all(classifiers, training_data, labels)

# print the classifiers
print(len(classifiers))
BagOfClassifiers.print_wrong_by_classifier(classifiers)

# filter out the classifiers that are dominated by another classifier
classifiers = BagOfClassifiers.eliminate_dominated_classifiers(classifiers)

# print the remaining classifiers after filtering
BagOfClassifiers.print_wrong_by_classifier(classifiers)

# run the boosting algorithm
numTrainingDataPts = len(training_data)
accuracies = []
strong_classifier_iter = Booster.generate_strong_classifier_iter(
    NUM_BOOSTING_ITERATIONS, numTrainingDataPts, classifiers
)
numIter = 0
for strong_classifier in strong_classifier_iter:
    numIter += 1
    # strong_classifier = Booster.generate_strong_classifier(
    #     numIter, numTrainingDataPts, classifiers
    # )
    accuracy = Booster.classify_training_data(strong_classifier, training_data, labels)
    accuracies.append((numIter, accuracy))
    print(f"accuracy={accuracy} with numIter={numIter}")

if SHOW_PLOT:
    plot()
