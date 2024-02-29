class FeatureVector:
    def __init__(self, *data):
        self.data = data


class Feature:
    def __init__(self, name: str):
        self.name = name

    def isBinary(self) -> bool:
        return isinstance(self, BinaryFeature)

    def isNumerical(self) -> bool:
        return isinstance(self, NumericalFeature)


class NumericalFeature(Feature):
    def __init__(self, name: str):
        super().__init__(name)


class BinaryFeature(Feature):
    def __init__(self, name: str):
        super().__init__(name)
