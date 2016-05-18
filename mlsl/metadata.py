class Metadata:
    """Data about the data."""

    def __init__(self,
                 features,
                 target,
                 path,
                 data=None,
                 mean=None,
                 std=None) -> None:
        self.data = data
        self.features = features
        self.target = target
        self.path = path
        self.mean = mean
        self.std = std
