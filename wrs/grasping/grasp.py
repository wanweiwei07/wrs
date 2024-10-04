import pickle


class Grasp(object):

    def __init__(self, ee_values=None, ac_pos=None, ac_rotmat=None):
        self.ee_values = ee_values
        self.ac_pos = ac_pos
        self.ac_rotmat = ac_rotmat

    def __str__(self):
        return "Grasp at " + repr(self.ac_pos) + "\n\t\t" + repr(self.ac_rotmat) + " with " + str(self.ee_values)


class GraspCollection(object):
    def __init__(self, end_effector=None, grasp_list=None):
        self.end_effector = end_effector
        if grasp_list is None:
            self._grasp_list = []
        else:
            self._grasp_list = grasp_list

    @classmethod
    def load_from_disk(cls, file_name="reference_grasp_collection.pickle"):
        with open(file_name, 'rb') as file:
            obj = pickle.load(file)
            if not isinstance(obj, cls):
                raise TypeError(f"Object in {file_name} is not an instance of {cls.__name__}")
            return obj

    def append(self, grasp):
        self._grasp_list.append(grasp)

    def pop(self, index=None):
        if index is None:
            return self._grasp_list.pop()
        return self._grasp_list.pop(index)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._grasp_list[index]
        elif isinstance(index, list):
            return [self._grasp_list[i] for i in index]
        else:
            raise Exception("Index type not supported.")

    def __setitem__(self, index, grasp):
        self._grasp_list[index] = grasp

    def __len__(self):
        return len(self._grasp_list)

    def __iter__(self):
        return iter(self._grasp_list)

    def __add__(self, other):
        if self.end_effector is other.end_effector:
            self._grasp_list += other._grasp_list
            return self
        else:
            raise ValueError("End effectors are not the same.")

    def __str__(self):
        out_str = "GraspCollection of: " + self.end_effector.name if self.end_effector is not None else "" + "\n"
        if len(self._grasp_list) == 0:
            out_str += "  No grasps in the collection.\n"
            return out_str
        for grasp in self._grasp_list:
            out_str += "  " + str(grasp) + "\n"
        return out_str

    def save_to_disk(self, file_name='reference_grasp_collection.pickle'):
        """
        :param reference_grasp_collection:
        :param file_name:
        :return:
        author: haochen, weiwei
        date: 20200104, 20240319
        """
        with open(file_name, 'wb') as file:
            pickle.dump(GraspCollection(grasp_list=self._grasp_list), file)
