from dataset.conveni_scene_generator_common import *


class XylitolBottleGroup(GridProductGroup):
    @property
    def flavors(self):
        return ["xylitol", "xylitol_7assort", "clorets", "xylitol_freshmint"]

    @property
    def number_of_objects(self):
        return [[2, 4], [2, 4], [1, 2]]

    @property
    def object_dimension(self):
        return [0.0686, 0.0710, 0.0896]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0343, -0.0355, 0], R=tf.euler.euler2mat(np.pi / 2, 0, -np.pi / 2), Z=[1, 1, 1])


class KinokonoyamaGroup(GridProductGroup):
    @property
    def flavors(self):
        return ["kinokonoyama"]

    @property
    def number_of_objects(self):
        return [[2, 6], [1, 3], [1, 2]]

    @property
    def object_dimension(self):
        return [0.0345, 0.1530, 0.0906]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0173, -0.0765, 0], R=tf.euler.euler2mat(np.pi / 2, 0, -np.pi / 2), Z=[1, 1, 1])


class CalorieMateGroup(GridProductGroup):
    @property
    def flavors(self):
        return ["calorie_mate_cheese", "calorie_mate_fruit"]

    @property
    def number_of_objects(self):
        return [[1, 2], [2, 4], [1, 4]]

    @property
    def object_dimension(self):
        return [0.1035, 0.1085, 0.0267]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0, -0.0543, 0.0134], R=tf.euler.euler2mat(0, 0, -np.pi / 2), Z=[1, 1, 1])


class JavaCurryGroup(GridProductGroup):
    @property
    def flavors(self):
        return ["java_curry_chukara", "vermont_curry_amakuchi", "vermont_curry_chukara"]

    @property
    def number_of_objects(self):
        return [[1, 2], [2, 5], [2, 4]]

    @property
    def object_dimension(self):
        return [0.1587, 0.0752, 0.0284]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0794, -0.0752, 0.0142], R=tf.euler.euler2mat(0, 0, 0), Z=[1, 1, 1])


class CannedIwashiGroup(GridProductGroup):
    @property
    def flavors(self):
        return ["canned_iwashi_kabayaki"]

    @property
    def number_of_objects(self):
        return [[1, 2], [2, 4], [2, 4]]

    @property
    def object_dimension(self):
        return [0.1071, 0.064, 0.0315]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0536, -0.032, 0], R=tf.euler.euler2mat(np.pi / 2, 0, np.pi), Z=[1, 1, 1])


class KaoSoapGroup(GridProductGroup):
    @property
    def flavors(self):
        return ["kao_soap"]

    @property
    def number_of_objects(self):
        return [[1, 2], [2, 4], [2, 3]]

    @property
    def object_dimension(self):
        return [0.0607, 0.0956, 0.0309]

    @property
    def model_frame(self):
        return tf.affines.compose(T=[0.0304, 0, 0.0155], R=tf.euler.euler2mat(0, 0, np.pi), Z=[1, 1, 1])


class DisplayPlanner(DisplayPlannerBase):
    def __init__(
        self,
        groups=[
            XylitolBottleGroup,
            KinokonoyamaGroup,
            CalorieMateGroup,
            JavaCurryGroup,
            CannedIwashiGroup,
            KaoSoapGroup,
        ],
        pos_x=0.57,
        pos_y_start=0.40,
        pos_y_end=-0.40,
        pos_z=1.165,
    ):
        super().__init__(groups, pos_x, pos_y_start, pos_y_end, pos_z)

    def find_pickable_object_in_group(self, group):
        def isin_approx(pos, poss, atol=0.01):
            return np.any([np.allclose(pos, x, atol=atol) for x in poss])

        pickable = []
        dim = group.object_dimension
        if (
            isinstance(group, CalorieMateGroup)
            or isinstance(group, KaoSoapGroup)
            or isinstance(group, CannedIwashiGroup)
            or isinstance(group, JavaCurryGroup)
        ):
            poss = [o[2][0] for o in group._plan]
            for name, object_id, (pos, quat), scale in group._plan:
                if isin_approx(pos + [0, 0, group.object_dimension[2]], poss):  ## another object is on top of this
                    continue
                if isin_approx(pos + [group.object_dimension[0], 0, 0], poss):  ## another object is on the back of this
                    continue
                if isin_approx(
                    pos + [-group.object_dimension[0], 0, 0], poss
                ):  ## another object is on the front of this
                    continue

                if pos[1] < 0.1 or pos[1] > 0.5:  # rough reacheablity test
                    continue

                pickable.append((group, name, object_id, (pos, quat), scale))

        return pickable


dp = DisplayPlanner(pos_x=0.57, pos_y_start=0.35, pos_y_end=-0.30, pos_z=1.165)
