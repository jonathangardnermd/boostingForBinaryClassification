from util.featureUtil import BinaryFeature, NumericalFeature, Feature, FeatureVector


class InputFeatureVector(FeatureVector):
    def __init__(
        self,
        evil: bool,
        emo: bool,
        transforms: bool,
        sparkly: bool,
        numRomanticInterests: int,
    ):
        super().__init__(evil, emo, transforms, sparkly, numRomanticInterests)


class InputData:
    def __init__(self, *args):
        self.features: list[Feature] = [
            BinaryFeature("Evil"),
            BinaryFeature("Emo"),
            BinaryFeature("Transforms"),
            BinaryFeature("Sparkly"),
            NumericalFeature("NumRomanticInterests"),
        ]

        self.trainingData: list[FeatureVector] = [
            FeatureVector(1, 0, 1, 0, 5),
            FeatureVector(0, 1, 1, 0, 5),
            FeatureVector(0, 1, 0, 1, 1),
            FeatureVector(0, 1, 0, 0, 3),
            FeatureVector(0, 1, 0, 0, 5),
            FeatureVector(1, 0, 1, 0, 5),
            FeatureVector(1, 0, 1, 0, 5),
            FeatureVector(0, 0, 1, 1, 1),
            FeatureVector(0, 1, 0, 0, 1),
            FeatureVector(0, 0, 0, 0, 5),
        ]

        self.labels: list[int] = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1]
