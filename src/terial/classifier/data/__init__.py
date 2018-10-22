from pathlib import Path


class RawDataset:
    def __init__(self, path, split_set):
        self.path: Path = path
        self.split_set = split_set

    def __len__(self):
        count = 0
        for client_dir in self.path.iterdir():
            for epoch_dir in client_dir.iterdir():
                count += len(
                    list((epoch_dir / self.split_set).glob('*.params.json')))
        return count

    def __iter__(self):
        for client_dir in self.path.iterdir():
            for epoch_dir in client_dir.iterdir():
                for path in (epoch_dir / self.split_set).glob('*.params.json'):
                    prefix = path.name.split('.')[0]
                    yield (epoch_dir / self.split_set, prefix)
