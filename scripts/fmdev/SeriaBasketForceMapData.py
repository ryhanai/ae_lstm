from TabletopForceMapData import *


class SeriaBasketRandomSceneDataset(TabletopRandomSceneDataset):
    """SeriaBasketRandomScene dataset.

    Args:
        data_type (string):        Set the data type (train/test) .
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
        root (string, optional):   Root directory of the data set, default is saved in the '~/epil/'.
        download (bool, optional): If True, downloads the dataset from the internet and
                                   puts it in root directory. If dataset is already downloaded, it is not
                                   downloaded again.
    """

    def __init__(
        self,
        data_type,
        minmax=[0.1, 0.9],
        fminmax=[1e-7, 1e-3],
        img_format="CWH",
        root_dir=Path(os.path.expanduser("~")) / "Dataset/forcemap/",
        # task_name="basket240511", ## need smoothing
        task_name="basket-filling230918",
        num_samples=2000,
        num_views=3,
        method="geometry-aware",  # basket-filling230918 only has "geometry-aware" smoothed force label
    ):
        super().__init__(data_type, minmax, fminmax, img_format, root_dir, task_name, num_samples, num_views, method)

    def _load_data(self):
        super()._load_data(split=[0.80, 0.98])

    def load_fmap(self, idx):
        scene_idx = self._ids[idx]
        fmap = pd.read_pickle(self._input_dir / f"force_zip{scene_idx:05}.pkl")
        fmap = fmap[:, :, :30].astype("float32")

        if self._method == 0 or self._method == 1:
            fmap = np.clip(fmap, self._force_bounds[0], self._force_bounds[1])
            fmap = np.log(fmap)  # force_raw (in log scale)
            fmap = fmap.transpose(2, 0, 1)
            fmap = self._normalization(fmap, np.log(self._force_bounds))
        else:
            dist_bounds = [-0.001, 0.02]
            fmap = np.clip(-fmap, dist_bounds[0], dist_bounds[1])
            fmap = fmap.transpose(2, 0, 1)
            fmap = self._normalization(fmap, dist_bounds)
        return fmap

    def reformat(self, d):
        # used for visualization with fmap_visualizer
        d = np.array(d)
        d = d.transpose(1, 2, 0)
        e = np.zeros((40, 40, 40))
        e[:, :, :30] = d
        return e
