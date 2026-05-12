import numpy as np
from torch.utils.data import Dataset
from feeders import tools


class Feeder(Dataset):
    def __init__(self,
                 data_path,
                 label_path=None,
                 p_interval=1,
                 split='train',
                 random_choose=False,
                 random_shift=False,
                 random_move=False,
                 random_rot=False,
                 window_size=-1,
                 normalization=False,
                 debug=False,
                 use_mmap=False,
                 bone=False,
                 vel=False,
                 upper=False,
                 num_point=70,
                 num_person=1):
        """
        MHR70 feeder for DeGCN/ST-GCN style framework + RGB.

        Input npz expected format:
          x_train: (N, Tmax, num_person*num_point*3)
          x_train_rgb: (N, 3, H, W)
          y_train: (N, num_class) one-hot
          x_test : (N, Tmax, num_person*num_point*3)
        x_test_rgb : (N, 3, H, W)
          y_test : (N, num_class) one-hot

        Output tensor per sample:
          data_numpy: (C, T, V, M)
          data_rgb: (3, H, W) 
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split

        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot

        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval

        self.bone = bone
        self.vel = vel
        self.upper = upper

        self.num_point = num_point
        self.num_person = num_person

        self.load_data()

        if normalization:
            self.get_mean_map()

    def load_data(self):
        npz_data = np.load(self.data_path)

        if self.split == 'train':
            self.data = npz_data['x_train']
            self.data_rgb = npz_data['x_train_rgb']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.data_rgb = npz_data['x_test_rgb']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError("data split only supports train/test")

        if self.debug:
            self.data = self.data[:100]
            self.data_rgb = self.data_rgb[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

        N, T, D = self.data.shape
        expected_dim = self.num_person * self.num_point * 3

        if D != expected_dim:
            raise ValueError(
                f"Input feature dim mismatch: got {D}, expected {expected_dim} "
                f"(num_person={self.num_person}, num_point={self.num_point})"
            )

        # (N,T,D) -> (N,T,M,V,C)
        self.data = self.data.reshape((N, T, self.num_person, self.num_point, 3))

        # -> (N,C,T,V,M)
        self.data = self.data.transpose(0, 4, 1, 3, 2).astype(np.float32)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape

        self.mean_map = data.mean(axis=2, keepdims=True) \
                            .mean(axis=4, keepdims=True) \
                            .mean(axis=0)

        self.std_map = data.transpose((0, 2, 4, 1, 3)) \
                           .reshape((N * T * M, C * V)) \
                           .std(axis=0) \
                           .reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])  # (C,T,V,M)
        data_rgb = np.array(self.data_rgb[index])  # (3,H,W)
        # print("data_numpy shape:", data_numpy.shape)
        # print("data_rgb shape:", data_rgb.shape)
        label = self.label[index]

        # valid frames count
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # crop/resize temporal length
        data_numpy = tools.valid_crop_resize(
            data_numpy,
            valid_frame_num,
            self.p_interval,
            self.window_size
        )

        # filter out upper body joints if needed
        if self.upper:
            from .bone_pairs import LOWER_BODY_INDEX
            all_indices = set(range(self.num_point))
            upper_indices = sorted(all_indices - set(LOWER_BODY_INDEX))
            data_numpy = data_numpy[:, :, upper_indices, :]

        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # bone modality
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            if self.upper:
                from .bone_pairs import mhr70_pairs_upper
                for v1, v2 in mhr70_pairs_upper:
                    bone_data_numpy[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]
            else:
                from .bone_pairs import mhr70_pairs
                for v1, v2 in mhr70_pairs:
                    bone_data_numpy[:, :, v1, :] = data_numpy[:, :, v1, :] - data_numpy[:, :, v2, :]

            data_numpy = bone_data_numpy

        # velocity modality
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, data_rgb, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)