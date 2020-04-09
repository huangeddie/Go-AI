class Policy:
    """
    Interface for all types of policies
    """

    def __init__(self, name, temp=None):
        self.name = name
        self.temp = temp
        self.pt_model = None

    def __call__(self, go_env, **kwargs):
        """
        :param go_env: Go environment
        :param step: the number of steps taken in the game so far
        :return: Action probabilities
        """
        pass

    def __str__(self):
        return "{} {}".format(self.__class__.__name__, self.name)
