from abc import ABC, abstractmethod


class MapBase(ABC):
    @abstractmethod
    def get_indices_for_ee_pose(self, tf_ee):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @abstractmethod
    def mark_reachable(self, map_indices):  # should it get a value? or an option for a value?
        pass

    @abstractmethod
    def is_reachable(self, map_indices):
        pass
