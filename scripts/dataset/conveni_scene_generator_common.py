import random
from abc import ABCMeta, abstractmethod
from functools import reduce

import numpy as np
import transforms3d as tf
from dataset.object_loader import ObjectInfo


class ProductGroup(metaclass=ABCMeta):
    def __init__(self, group_id, missing_items=True):
        self._group_id = group_id
        self._missing_items = missing_items
        self._width = 0.0
        self._plan = []

    @abstractmethod
    def plan(self, base_frame):
        pass

    def __repr__(self):
        return f"{self.__class__}:\n {reduce(lambda x, y: f'{x}, {y}', self._plan)}"

    def __str__(self):
        return self.__repr__()

    def get_width(self):
        return self._width

    def get_arrangement(self):
        return self._plan


class GridProductGroup(ProductGroup):
    @property
    @abstractmethod
    def flavors(self):
        pass

    @property
    @abstractmethod
    def number_of_objects(self):
        pass

    @property
    @abstractmethod
    def object_dimension(self):
        """
        sizes of objects in the pose where it is placed"
        """

    @property
    @abstractmethod
    def model_frame(self):
        """
        transform from object frame to origine defined in 3D model
        """

    def plan(self, base_frame, n_rows=None, n_columns=None, n_levels=None):
        result = []
        if n_rows == None:
            n_rows = np.random.randint(self.number_of_objects[0][0], self.number_of_objects[0][1])
        if n_columns == None:
            n_columns = np.random.randint(self.number_of_objects[1][0], self.number_of_objects[1][1])
        if n_levels == None:
            n_levels = np.random.randint(self.number_of_objects[2][0], self.number_of_objects[2][1])

        product_name = random.choice(self.flavors)

        n_objects = 0
        for i in range(n_rows):
            for j in range(n_columns):
                for k in range(n_levels):
                    if self._missing_items and k >= 1 and np.random.randint(10) < 3:
                        break  # finish piling up this position

                    # transform from group base frame to object frame
                    object_frame = tf.affines.compose(
                        T=np.asarray(self.object_dimension) * np.array([i, -j, k]),
                        R=tf.euler.euler2mat(0, 0, 0),
                        Z=[1, 1, 1],
                    )
                    pose = base_frame @ object_frame @ self.model_frame
                    pos, R, _, _ = tf.affines.decompose(pose)
                    ori = tf.quaternions.mat2quat(R)
                    result.append((product_name, (self._group_id, n_objects), (pos, ori), [1.0, 1.0, 1.0]))
                    n_objects += 1

        self._plan = result
        self._width = self.object_dimension[1] * n_columns


class GridInBasketGroup(GridProductGroup):
    def plan(self, base_frame):
        n_columns = np.random.randint(self.number_of_objects[1][0], self.number_of_objects[1][1])

        offset_base_frame = base_frame.copy()
        offset_base_frame[0, 3] += 0.015
        offset_base_frame[1, 3] -= 0.015
        super().plan(base_frame=offset_base_frame, n_columns=n_columns)

        basket_model_width = 0.26
        basket_width = self._width + 0.03
        basket_frame = tf.affines.compose(
            T=np.asarray([basket_model_width / 4, -basket_width / 2, 0]),
            R=tf.euler.euler2mat(np.pi / 2, 0, 0),
            Z=[1, 1, 1],
        )
        pos, R, _, _ = tf.affines.decompose(base_frame @ basket_frame)

        self._plan.append(
            (
                "cardboard_food_tray_1510",
                (self._group_id, 0),
                (pos, tf.quaternions.mat2quat(R)),
                [0.005, 0.01, 0.01 * basket_width / basket_model_width],
            )
        )

        self._width = basket_width


class DisplayPlannerBase:
    def __init__(self, groups, pos_x=0.58, pos_y_start=0.80, pos_y_end=-0.50, pos_z=1.165):
        self._product_info = ObjectInfo("ycb_conveni_v1", split="all")
        self._groups = groups
        self._pos_x = pos_x
        self._pos_y_start = pos_y_start
        self._pos_y_end = pos_y_end
        self._pos_z = pos_z

    def display_plan(
        self,
    ):
        while True:
            self._group_id = 0
            self._plan = []
            group_base_frame = tf.affines.compose(
                T=[self._pos_x, self._pos_y_start, self._pos_z], R=tf.euler.euler2mat(0, 0, 0), Z=[1, 1, 1]
            )
            while group_base_frame[1, 3] > self._pos_y_end:
                g = self.sample_group(group_base_frame)
                self._plan.append(g)
                group_base_frame[1, 3] -= g.get_width()
                # randomly make open space between groups
                if np.random.randint(8) == 0:
                    group_base_frame[1, 3] -= np.random.uniform(0.03, 0.2)

            target_objects = self.find_pickable_object()
            if len(target_objects) > 0:
                target_object = random.choice(target_objects)
                break

            print("REPLAN")

        return self._plan, target_object

    def sample_group(self, base_frame):
        g = np.random.choice(self._groups)
        g_inst = g(self._group_id)
        g_inst.plan(base_frame)
        self._group_id += 1
        return g_inst

    def find_pickable_object(self):
        pickable_objects = []
        for group in self._plan:
            pickable_objects.extend(self.find_pickable_object_in_group(group))
        return pickable_objects
