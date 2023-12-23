import os
import yaml

class ObjectsInfo:
    def __init__(self, object_dir, config_dir, dataset='ycb_conveni'):
        self.load_config(object_dir, config_dir, dataset)

    def load_config(self, object_dir, config_dir, dataset):
        with open(os.path.join(config_dir, f'dataset_{dataset}.yaml')) as f:
            dataset_def = yaml.safe_load(f)['train']

        with open(os.path.join(f'{config_dir}', 'objects.yaml')) as f:
            obj_def = yaml.safe_load(f)

        self._info = {}
        for name in dataset_def['ycb']:
            prop = obj_def['ycb'][name]
            self._info[name] = {
                'mass': prop['mass'],
                'usd_file': f'{object_dir}/ycb/{name}/google_16k/textured/textured.usd'
            }
        for name in dataset_def['conveni']:
            prop = obj_def['conveni'][name]
            self._info[name] = {
                'mass': prop['mass'],
                'usd_file': f'{object_dir}/conveni/{name}/{name}/{name}.usd'
            }

    def usd_file(self, name):
        return self._info[name]['usd_file']

    def mass(self, name):
        return self._info[name]['mass']

    def names(self):
        return list(self._info.keys())

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        names = self.names()
        if self._i >= len(names):
            raise StopIteration
        name = names[self._i]
        self._i += 1
        return name, self.usd_file(name), self.mass(name)


if __name__ == '__main__':
    object_dir = f'{os.environ["HOME"]}/Dataset/ycb_conveni'
    config_dir = f'{os.environ["HOME"]}/Program/moonshot/ae_lstm/specification/config'
    conf = ObjectsInfo(object_dir, config_dir, 'ycb_conveni_v1')