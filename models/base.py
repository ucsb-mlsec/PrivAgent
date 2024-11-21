from together.types import abstract


class BaseModel:
    @abstract
    def train(self):
        pass

    def evaluate(self):
        pass
