import csv
import json
import os

from terial import config


def get_synset_dir(synset_id):
    return config.SHAPENET_CORE_DIR / synset_id


def get_synset_models(synset_id):
    synset_meta = load_metadata(synset_id)
    synset_dir = get_synset_dir(synset_id)
    models = []
    for model_id in os.listdir(synset_dir):
        path = os.path.join(synset_dir, model_id)
        if os.path.isdir(path):
            models.append(Model(synset_meta[model_id], model_id, path))
    return sorted(models, key=lambda m: m.model_id)


def get_model(synset_id, model_id):
    synset_meta = load_metadata(synset_id)
    synset_dir = get_synset_dir(synset_id)
    path = os.path.join(synset_dir, model_id)
    return Model(synset_meta[model_id], model_id, path)


def _make_synset_ids_by_name(data):
    synsets_by_name = {}
    for item in data:
        for name in item['name'].split(','):
            if name not in synsets_by_name:
                synsets_by_name[name] = []
            synsets_by_name[name].append(item['synsetId'])
            synsets_by_name[name].extend(item['children'])
    return synsets_by_name


def load_metadata(synset_id):
    path = os.path.join(config.SHAPENET_META_DIR,
                        '{}.csv'.format(synset_id))
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    rows.sort(key=lambda x: x['fullId'].split('.')[-1])
    return {m['fullId'].split('.')[-1]: m for m in rows}


def load_all_metadata():
    metadata = {}
    for path in config.SHAPENET_META_DIR.glob('*.csv'):
        metadata.update(load_metadata(path.name.split('.')[0]))
    return metadata


class Model:
    metadata = load_all_metadata()

    @classmethod
    def from_id(cls, shape_id):
        m = Model.metadata[shape_id]
        synset_ids = m['wnsynset'].split(',')
        shape_path = None
        for synset_id in synset_ids:
            shape_path = get_synset_dir(synset_id) / shape_id
            if shape_path.exists():
                break

        if shape_path is None:
            raise ValueError('Cannot find shape path')

        return cls(m, shape_id, str(shape_path))


    def __init__(self, meta, model_id, path):
        assert model_id == meta['fullId'].split('.')[-1]
        self.full_id = meta['fullId']
        self.model_id = model_id
        self.synset_id = meta['wnsynset'].split(',')[0]
        self.path = path

    def has_uvs(self):
        uv_mapped_path = os.path.join(self.path, 'models/model_uvmapped.obj')
        return os.path.exists(uv_mapped_path)

    @property
    def obj_path(self):
        uv_mapped_path = os.path.join(self.path, 'models/model_uvmapped.obj')
        return uv_mapped_path

    @property
    def orig_obj_path(self):
        orig_path = os.path.join(self.path, 'models/model_normalized.obj')
        return orig_path

    @property
    def mtl_path(self):
        orig_path = os.path.join(self.path, 'models/model_normalized.mtl')
        return orig_path
        # uv_mapped_path = os.path.join(self.path, 'models/model_uvmapped.mtl')
        # if not os.path.exists(uv_mapped_path):
        #     logger.warning("UV mapped .mtl file does not exist.")
        #     orig_path = os.path.join(self.path, 'models/model_normalized.mtl')
        #     return orig_path
        # return uv_mapped_path

    @property
    def rend_subdir(self):
        return os.path.join(self.synset_id, self.model_id)


class Taxonomy:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.synset_id_list = [s['synsetId'] for s in self.data]
        self.synset_ids_by_name = _make_synset_ids_by_name(self.data)
        self.synset_name_by_id = {s['synsetId']: s['name'] for s in self.data}

    def name_to_id(self, synset_name):
        return self.synset_ids_by_name[synset_name][0]

    def id_to_name(self, synset_id):
        return self.synset_name_by_id[synset_id]


taxonomy = Taxonomy(config.SHAPENET_TAXONOMY_PATH)
