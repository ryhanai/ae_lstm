import colorsys
import rviz_client
import numpy as np
import scipy.linalg
import forcemap


class ForceDistributionViewer:
    """
        Singleton pattern
        Duplicated instantiation causes the error of ROS node intialization
    """

    _unique_instance = None

    def __new__(cls):
        raise NotImplementationError('Cannot generate instance by constructor')

    @classmethod
    def __internal_new__(cls):
        return super().__new__(cls)

    @classmethod
    def get_instance(cls):
        if not cls._unique_instance:
            cls._unique_instance = cls.__internal_new__()
            cls.rviz_client = rviz_client.RVizClient()
        return cls._unique_instance

    # def publish_bin_state(self, bin_state, positions, fvals, draw_fmap, draw_force_gradient):
    #     self.rviz_client.delete_all()
    #     self.draw_bin()
    #     if bin_state is not None:
    #         self.draw_objects(bin_state)
    #     if draw_fmap:
    #         self.draw_force_distribution(positions, fvals)
    #     if draw_force_gradient:
    #         self.draw_force_gradient(positions, fvals)
    #     self.rviz_client.show()

    def publish_bin_state(self, bin_state, fmap, draw_fmap, draw_force_gradient):
        self.rviz_client.delete_all()
        self.draw_bin()

        positions = fmap.get_positions()
        fvals = fmap.get_values()

        if bin_state is not None:
            self.draw_objects(bin_state)
        if draw_fmap:
            self.draw_force_distribution(positions, fvals)
        if draw_force_gradient:
            self.draw_force_gradient(positions, fvals)
        self.rviz_client.show()

    def draw_bin(self):
        self.rviz_client.draw_mesh("package://fmap_visualizer/meshes_extra/seria_basket.dae",
                                   ((0, 0, 0.73), (0, 0, 0.70711, 0.70711)),
                                   (0.5, 0.5, 0.5, 0.2))

    def draw_objects(self, bin_state):
        for object_state in bin_state:
            name, pose = object_state
            self.rviz_client.draw_mesh("package://fmap_visualizer/meshes/{}/google_16k/textured.dae".format(name),
                                       pose,
                                       (0.5, 0.5, 0.5, 0.3))

    def draw_force_distribution(self, positions, fvals):
        fvals = fvals.flatten()
        fmax = np.max(fvals)
        fmin = np.min(fvals)
        points = []
        rgbas = []
        if fmax - fmin > 1e-3:
            # std_fvals = np.clip(5.0 * (fvals - fmin) / (fmax - fmin), 0.0, 1.0)
            std_fvals = (fvals - fmin) / (fmax - fmin)
            for (x, y, z), f in zip(positions, std_fvals):
                if f > 0.08:
                    points.append([x, y, z])
                    r, g, b = colorsys.hsv_to_rgb(1./3 * (1-f), 1, 1)
                    rgbas.append([r, g, b, 1])
        self.rviz_client.draw_points(points, rgbas)

    def draw_vector_field(self, positions, values, scale=0.3):
        self.rviz_client.draw_arrows(positions, positions+values*scale)

    def draw_force_gradient(self, positions, fvals, scale=0.3, threshold=0.008):
        gxyz = np.gradient(- fvals)
        g_vecs = np.column_stack([g.flatten() for g in gxyz])
        pos_val_pairs = [(p, g) for (p, g) in zip(positions, g_vecs) if scipy.linalg.norm(g) > threshold]
        positions, values = zip(*pos_val_pairs)
        self.draw_vector_field(np.array(positions), np.array(values), scale=scale)