from abc import ABC, abstractmethod

class Simulation(ABC):

    @abstractmethod
    def run(self):
        pass