import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from rdkit import Chem

from torch_geometric.data import Data, DataLoader, InMemoryDataset, download_url, extract_zip

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}


class QM93D(InMemoryDataset):
    r"""
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for :obj:`QM9` dataset 
        which is from `"Quantum chemistry structures and properties of 134 kilo molecules" <https://www.nature.com/articles/sdata201422>`_ paper.
        It connsists of about 130,000 equilibrium molecules with 12 regression targets: 
        :obj:`mu`, :obj:`alpha`, :obj:`homo`, :obj:`lumo`, :obj:`gap`, :obj:`r2`, :obj:`zpve`, :obj:`U0`, :obj:`U`, :obj:`H`, :obj:`G`, :obj:`Cv`.
        Each molecule includes complete spatial information for the single low energy conformation of the atoms in the molecule.

        .. note::
            Based on the code of `QM9 in Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/qm9.html#QM9>`_.

    
        Args:
            root (string): the dataset folder will be located at root/qm9.
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)

        Example:
        --------

        >>> dataset = QM93D()
        >>> target = 'mu'
        >>> dataset.data.y = dataset.data[target]
        >>> split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
        >>> train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> data = next(iter(train_loader))
        >>> data
        Batch(Cv=[32], G=[32], H=[32], U=[32], U0=[32], alpha=[32], batch=[579], gap=[32], homo=[32], lumo=[32], mu=[32], pos=[579, 3], ptr=[33], r2=[32], y=[32], z=[579], zpve=[32])

        Where the attributes of the output data indicates:
    
        * :obj:`z`: The atom type.
        * :obj:`pos`: The 3D position for atoms.
        * :obj:`y`: The target property for the graph (molecule).
        * :obj:`batch`: The assignment vector which maps each node to its respective graph identifier and can help reconstructe single graphs
    """
    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):

        self.raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
        self.raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
        self.folder = osp.join(root, 'qm9')

        super(QM93D, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self):
        return 'qm9_pyg.pt'

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'),
                    osp.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        
        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
            posc = pos - pos.mean(dim=0)

            atomic_number = []
            
            for atom in mol.GetAtoms():
                atomic_number.append(atom.GetAtomicNum())
                
            z = torch.tensor(atomic_number, dtype=torch.long)

            data = Data(
                z=z,
                pos=pos,
                posc=posc,
                y=y[i].unsqueeze(0),
                mu=y[i][0], alpha=y[i][1], homo=y[i][2], lumo=y[i][3], gap=y[i][4], r2=y[i][5], zpve=y[i][6], U0=y[i][7], U=y[i][12], H=y[i][13], G=y[i][14], Cv=y[i][15]
            )

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = QM93D(root='dataset/')
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    target = 'mu'
    dataset.data.y = dataset.data[target]
    print(dataset.data.y.shape)
    print(dataset.data.y)
    print(dataset.data.mu)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    data = next(iter(train_loader))
    print(data)
