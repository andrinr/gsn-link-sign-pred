class Feature():
    """
    Feature class to store the feature size and hidden size of a feature.
    """
    def __init__(
        self,
        feature_size : int,
        hidden_size : int
    ):
        self.feature_size = feature_size
        self.hidden_size = hidden_size