from abc import ABC, abstractmethod
from typing import Tuple


class SimulatorAgents(ABC):
    @abstractmethod
    def get_variant_types(self) -> Tuple[list, list]:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, time_step: int):
        pass

    @abstractmethod
    def get_action_list(self, observation: dict):
        pass

    @abstractmethod
    def get_reset_options(self) -> dict:
        pass
