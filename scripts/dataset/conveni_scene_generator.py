import numpy as np
import transforms3d as tf
from abc import ABCMeta, abstractmethod
from functools import reduce
from object_loader import ObjectInfo


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

        product_name = np.random.choice(self.flavors)

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
                        Z=[1, 1, 1])
                    pose = base_frame @ object_frame @ self.model_frame
                    pos, R, _, _ = tf.affines.decompose(pose)
                    ori = tf.quaternions.mat2quat(R)
                    result.append((product_name, (self._group_id, n_objects), (pos, ori), [1., 1., 1.]))
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
            T=np.asarray([basket_model_width/4, -basket_width/2, 0]),
            R=tf.euler.euler2mat(np.pi/2, 0, 0),
            Z=[1, 1, 1]
        )
        pos, R, _, _ =tf.affines.decompose(base_frame @ basket_frame)

        self._plan.append(
            ('cardboard_food_tray_1510',
             (self._group_id, 0),
             (pos, tf.quaternions.mat2quat(R)),
             [0.005, 0.01, 0.01 * basket_width / basket_model_width])
             )
        
        self._width = basket_width


class CupNoodleGroup(GridProductGroup):
    @property
    def flavors(self):
        return ['cupnoodle_curry', '2nd_cupnoodle']
    
    @property
    def number_of_objects(self):
        return [[2,4], [2,4], [1,3]]
    
    @property
    def object_dimension(self):
        return [0.0969, 0.0962, 0.1085]
    
    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0485, -0.0481, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class XylitolBottleGroup(GridProductGroup):
    @property
    def flavors(self):
        return ['xylitol', 'xylitol_7assort', 'clorets', 'xylitol_freshmint']
    
    @property
    def number_of_objects(self):
        return [[2,4], [2,4], [1,2]]
    
    @property
    def object_dimension(self):
        return [0.0686, 0.0710, 0.0896]
    
    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0343, -0.0355, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class KinokonoyamaGroup(GridProductGroup):
    @property
    def flavors(self):
        return ['kinokonoyama']
    
    @property
    def number_of_objects(self):
        return [[2,6], [1,3], [1,2]]
    
    @property
    def object_dimension(self):
        return [0.0345, 0.1530, 0.0906]    
    
    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0173, -0.0765, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class CalorieMateGroup(GridProductGroup):
    @property    
    def flavors(self):
        return ['calorie_mate_cheese', 'calorie_mate_fruit']
    
    @property
    def number_of_objects(self):
        return [[1,3], [2,4], [1,4]]
    
    @property
    def object_dimension(self):
        return [0.1035, 0.1085, 0.0267]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0, -0.0543, 0.0134],
                                  R=tf.euler.euler2mat(0, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class ChocoballGroup(GridInBasketGroup):
    @property
    def flavors(self):
        return ['chocoball']    

    @property
    def number_of_objects(self):
        return [[2,5], [1,4], [1,2]]
    
    @property
    def object_dimension(self):
        return [0.0211, 0.0486, 0.0976]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0106, -0.0243, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class GingerTubeGroup(GridInBasketGroup):
    @property
    def flavors(self):
        return ['ginger_tube', 'kizami_aojiso_tube', 'honwasabi_tube']    

    @property
    def number_of_objects(self):
        return [[2,4], [2,5], [1,2]]
    
    @property
    def object_dimension(self):
        return [0.0329, 0.0411, 0.1477]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0165, -0.0206, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class JavaCurryGroup(GridProductGroup):
    @property    
    def flavors(self):
        return ['java_curry_chukara', 'vermont_curry_amakuchi', 'vermont_curry_chukara']
    
    @property
    def number_of_objects(self):
        return [[1,2], [2,5], [2,4]]
    
    @property
    def object_dimension(self):
        return [0.1587, 0.0752, 0.0284]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0794, -0.0752, 0.0142],
                                  R=tf.euler.euler2mat(0, 0, 0),
                                  Z=[1, 1, 1])

class SoysauceGroup(GridProductGroup):
    """
    This object might be unstable
    """
    @property    
    def flavors(self):
        return ['kikkoman_soysauce_gennenn', 'kikkoman_soysauce'] 
    
    @property
    def number_of_objects(self):
        return [[2,4], [2,5], [1,2]]
    
    @property
    def object_dimension(self):
        return [0.0614, 0.0634, 0.1781]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0307, -0.0317, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class CannedIwashiGroup(GridProductGroup):
    @property    
    def flavors(self):
        return ['canned_iwashi_kabayaki'] 
    
    @property
    def number_of_objects(self):
        return [[1,3], [2,4], [2,4]]
    
    @property
    def object_dimension(self):
        return [0.1071, 0.064, 0.0315]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0536, -0.032, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, np.pi),
                                  Z=[1, 1, 1])

class ClinicaHamigakiGroup(GridProductGroup):
    @property    
    def flavors(self):
        return ['clinica_ad_hamigaki', 'clinica_advantage', 'gum_paste', 'nonio_strong_energy', 'nonio_mintpaste']
    
    @property
    def number_of_objects(self):
        return [[2,5], [1,3], [1,2]]
    
    @property
    def object_dimension(self):
        return [0.0379, 0.0594, 0.1613]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0190, -0.0297, 0],
                                  R=tf.euler.euler2mat(np.pi/2, 0, -np.pi/2),
                                  Z=[1, 1, 1])

class KaoSoapGroup(GridProductGroup):
    @property    
    def flavors(self):
        return ['kao_soap']
    
    @property
    def number_of_objects(self):
        return [[2,4], [2,4], [2,3]]
    
    @property
    def object_dimension(self):
        return [0.0607, 0.0956, 0.0309]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0304, 0, 0.0155],
                                  R=tf.euler.euler2mat(0, 0, np.pi),
                                  Z=[1, 1, 1])


class DisplayPlanner:
    def __init__(self):
        self._product_info = ObjectInfo("ycb_conveni_v1", split='all')
        self._groups = [
            CupNoodleGroup, XylitolBottleGroup, KinokonoyamaGroup, CalorieMateGroup, 
            ChocoballGroup, GingerTubeGroup,
            JavaCurryGroup,
            CannedIwashiGroup, ClinicaHamigakiGroup, KaoSoapGroup
        ]

    def display_plan(self, pos_x=0.60, pos_y_start=0.80, pos_y_end=-0.50, pos_z=1.165):
        self._group_id = 0
        self._plan = []
        group_base_frame = tf.affines.compose(T=[pos_x, pos_y_start, pos_z], 
                                              R=tf.euler.euler2mat(0, 0, 0),
                                              Z=[1, 1, 1])
        while group_base_frame[1, 3] > pos_y_end:
            g = self.sample_group(group_base_frame)
            self._plan.append(g)
            group_base_frame[1, 3] -= g.get_width()
            # randomly make open space between groups
            if np.random.randint(8) == 0:
                group_base_frame[1, 3] -= np.random.uniform(0.03, 0.2)

        print(self._plan)
        return self._plan

    def sample_group(self, base_frame):
        g = np.random.choice(self._groups)
        g_inst = g(self._group_id)
        g_inst.plan(base_frame)
        self._group_id += 1
        return g_inst

