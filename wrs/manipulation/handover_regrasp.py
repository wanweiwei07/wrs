class HandoverPlanner(object):
    def __init__(self, robot_a, robot_b, obj_cmodel,
                 reference_grasp_collection_a, reference_grasp_collection_b):
        self.obj_cmodel = obj_cmodel
        self.robots = [robot_a, robot_b]
        self.reference_grasp_collections = [reference_grasp_collection_a, reference_grasp_collection_b]

    def add_hopg_collection_from_disk(self, file_name):