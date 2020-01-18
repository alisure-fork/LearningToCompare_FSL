import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


##############################################################################################################


_MEAN_PIXEL = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
_STD_PIXEL = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]


class ClassBalancedSampler(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]]
                     for j in range(self.num_cl)]
            pass

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
            pass

        return iter(batch)

    def __len__(self):
        return 1

    pass


class ClassBalancedSamplerTest(Sampler):
    """ Samples 'num_inst' examples each from 'num_cl' pools of examples of size 'num_per_class' """

    def __init__(self, num_cl, num_inst, shuffle=True):
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle
        pass

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
            pass

        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
                pass
            pass

        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

    pass


class MiniImageNetTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num, dhc_train_name=None):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        self.dhc_train_name = dhc_train_name

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        self.train_roots = []
        self.test_roots = []
        samples = dict()
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
            pass

        self.train_labels = [labels[os.path.split(x)[0]] for x in self.train_roots]
        self.test_labels = [labels[os.path.split(x)[0]] for x in self.test_roots]
        self.dhc_train_ids = [self.dhc_train_name.index(x) for x in self.train_roots] if self.dhc_train_name else None
        self.dhc_test_ids = [self.dhc_train_name.index(x) for x in self.test_roots] if self.dhc_train_name else None
        pass

    @staticmethod
    def folders(data_root):
        train_folder = os.path.join(data_root, "train")
        val_folder = os.path.join(data_root, "val")
        test_folder = os.path.join(data_root, "test")

        folders_train = [os.path.join(train_folder, label) for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]
        folders_val = [os.path.join(val_folder, label) for label in os.listdir(val_folder)
                       if os.path.isdir(os.path.join(val_folder, label))]
        folders_test = [os.path.join(test_folder, label) for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))]

        random.seed(1)
        random.shuffle(folders_train)
        random.shuffle(folders_val)
        random.shuffle(folders_test)

        dhc_train_name, dhc_val_name, dhc_test_name = [], [], []

        for folder in folders_train:
            dhc_train_name.extend([os.path.join(folder, name) for name in os.listdir(folder)])
        for folder in folders_val:
            dhc_val_name.extend([os.path.join(folder, name) for name in os.listdir(folder)])
        for folder in folders_test:
            dhc_test_name.extend([os.path.join(folder, name) for name in os.listdir(folder)])

        random.shuffle(dhc_train_name)
        random.shuffle(dhc_val_name)
        random.shuffle(dhc_test_name)

        return folders_train, folders_val, folders_test, dhc_train_name, dhc_val_name, dhc_test_name

    pass


class MiniImageNet(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.dhc_ids = self.task.dhc_train_ids if self.split == 'train' else self.task.dhc_test_ids
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        dhc_id = self.dhc_ids[idx] if self.dhc_ids else -1
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, dhc_id

    @staticmethod
    def get_data_loader_2(task, num_per_class=1, split='train', sampler_test=False, shuffle=False):
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if split == 'train':
            dataset = MiniImageNet(task, split=split, transform=transform_train)
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            dataset = MiniImageNet(task, split=split, transform=transform_test)
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    @staticmethod
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False):
        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)
        transform_train = transforms.Compose([transforms.RandomCrop(84, padding=8), transforms.RandomHorizontalFlip(),
                                              lambda x: np.asarray(x), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([lambda x: np.asarray(x), transforms.ToTensor(), normalize])

        if split == 'train':
            dataset = MiniImageNet(task, split=split, transform=transform_train)
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            dataset = MiniImageNet(task, split=split, transform=transform_test)
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass


class DHCMiniImageNet(Dataset):

    def __init__(self, dhc_data_name, transform=None):
        self.dhc_data_name = dhc_data_name
        _file_class = [os.path.basename(os.path.split(file_name)[0]) for file_name in self.dhc_data_name]
        _class_name = sorted(set(_file_class))
        self.train_label = [_class_name.index(file_class_name) for file_class_name in _file_class]
        self.transform = transform
        pass

    def __len__(self):
        return len(self.dhc_data_name)

    def __getitem__(self, idx):
        image_root = self.dhc_data_name[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        label = self.train_label[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, idx

    @staticmethod
    def get_data_loader(dhc_data_name, batch_size=16, shuffle=False):
        normalize = transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL)
        transform_test = transforms.Compose([lambda x: np.asarray(x), transforms.ToTensor(), normalize])

        dataset = DHCMiniImageNet(dhc_data_name, transform=transform_test)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    pass


##############################################################################################################


class CNNEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU())
        pass

    def forward(self, x, is_summary=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out4, {"x": x, "out1": out1, "out2": out2, "out3": out3, "out4": out4} if is_summary else None

    pass


class RelationNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(64, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Linear(64 * 3 * 3, 8)
        self.fc2 = nn.Linear(8, 1)
        pass

    def forward(self, x, is_summary=False):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out = out2.view(out2.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out, {"x": x, "out1": out1, "out2": out2} if is_summary else None

    pass


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
        pass

    def forward(self, x, dim=1):
        norm = x.pow(self.power).sum(dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

    pass


class DHCModel(nn.Module):

    def __init__(self, in_dim, out_dim, linear_bias=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer1 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(in_dim, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=0),
                                    nn.BatchNorm2d(in_dim, momentum=1, affine=True), nn.ReLU(), nn.MaxPool2d(2))
        self.linear_1 = nn.Linear(in_dim, out_dim, bias=linear_bias)
        self.l2norm = Normalize(2)
        pass

    def forward(self, x, is_summary=False):
        # out = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out_logits = self.linear_1(out)
        out_l2norm = self.l2norm(out_logits)
        out_sigmoid = torch.sigmoid(out_logits)

        return out_logits, out_l2norm, out_sigmoid

    pass


class DHCLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion_no = nn.CrossEntropyLoss()
        pass

    def forward(self, out, targets):
        loss_1 = self.criterion_no(out, targets)
        return loss_1

    pass


class DHCProduceClass(object):

    def __init__(self, n_sample, out_dim, ratio=1.0):
        super().__init__()
        self.out_dim = out_dim
        self.n_sample = n_sample
        self.class_per_num = self.n_sample // self.out_dim * ratio
        self.count = 0
        self.count_2 = 0
        self.class_num = np.zeros(shape=(self.out_dim, ), dtype=np.int)
        self.classes = np.zeros(shape=(self.n_sample, ), dtype=np.int)
        pass

    def reset(self):
        self.count = 0
        self.count_2 = 0
        self.class_num *= 0
        pass

    def cal_label(self, out, indexes):
        top_k = out.data.topk(self.out_dim, dim=1)[1].cpu()
        indexes_cpu = indexes.cpu()

        batch_size = top_k.size(0)
        class_labels = np.zeros(shape=(batch_size,), dtype=np.int)

        for i in range(batch_size):
            for j_index, j in enumerate(top_k[i]):
                if self.class_per_num > self.class_num[j]:
                    class_labels[i] = j
                    self.class_num[j] += 1
                    self.count += 1 if self.classes[indexes_cpu[i]] != j else 0
                    self.classes[indexes_cpu[i]] = j
                    self.count_2 += 1 if j_index != 0 else 0
                    break
                pass
            pass
        pass

    def get_label(self, indexes):
        return torch.tensor(self.classes[indexes.cpu()]).long().cuda()

    pass


class KNN(object):

    @staticmethod
    def knn(epoch, feature_encoder, dhc_model, low_dim, train_loader, k, t=0.1):

        with torch.no_grad():

            def _cal(_labels, _dist, _train_labels, _retrieval_1_hot, _top1, _top5, _max_c):
                # ---------------------------------------------------------------------------------- #
                _batch_size = _labels.size(0)
                _yd, _yi = _dist.topk(k+1, dim=1, largest=True, sorted=True)
                _yd, _yi = _yd[:, 1:], _yi[:, 1:]
                _candidates = train_labels.view(1, -1).expand(_batch_size, -1)
                _retrieval = torch.gather(_candidates, 1, _yi)

                _retrieval_1_hot.resize_(_batch_size * k, _max_c).zero_()
                _retrieval_1_hot = _retrieval_1_hot.scatter_(1, _retrieval.view(-1, 1), 1).view(_batch_size, -1, _max_c)
                _yd_transform = _yd.clone().div_(t).exp_().view(_batch_size, -1, 1)
                _probs = torch.sum(torch.mul(_retrieval_1_hot, _yd_transform), 1)
                _, _predictions = _probs.sort(1, True)
                # ---------------------------------------------------------------------------------- #

                _correct = _predictions.eq(_labels.data.view(-1, 1))

                _top1 += _correct.narrow(1, 0, 1).sum().item()
                _top5 += _correct.narrow(1, 0, 5).sum().item()
                return _top1, _top5, _retrieval_1_hot

            n_sample = train_loader.dataset.__len__()
            out_memory = torch.zeros(n_sample, low_dim).t().cuda()
            train_labels = torch.LongTensor(train_loader.dataset.train_label).cuda()
            max_c = train_labels.max() + 1

            out_list = []
            for batch_idx, (inputs, labels, indexes) in enumerate(train_loader):
                sample_features, _ = feature_encoder(inputs.cuda())  # 5x64*19*19
                _, out_l2norm, _ = dhc_model(sample_features)
                out_list.append([out_l2norm, labels.cuda()])
                out_memory[:, batch_idx * inputs.size(0):(batch_idx + 1) * inputs.size(0)] = out_l2norm.data.t()
                pass

            top1, top5, total = 0., 0., 0
            retrieval_one_hot = torch.zeros(k, max_c).cuda()  # [200, 10]
            for out in out_list:
                dist = torch.mm(out[0], out_memory)
                total += out[1].size(0)
                top1, top5, retrieval_one_hot = _cal(out[1], dist, train_labels, retrieval_one_hot, top1, top5, max_c)
                pass

            Tools.print("Test 1 {} Top1={:.2f} Top5={:.2f}".format(epoch, top1 * 100. / total, top5 * 100. / total))
            return top1 / total, top5 / total

        pass

    pass


##############################################################################################################


class Runner(object):

    def __init__(self, model_name, feature_encoder, relation_network, dhc_model, compare_fsl_fn,
                 train_episode=300000, data_root='/mnt/4T/Data/miniImagenet', summary_dir=None):
        self.class_num = 5
        self.sample_num_per_class = 1
        self.batch_num_per_class = 15

        self.train_episode = train_episode  # 500000
        self.val_episode = 600
        self.test_avg_num = 10
        self.test_episode = 600

        self.learning_rate = 0.001

        self.print_freq = 500
        self.val_freq = 5000  # 5000
        self.best_accuracy = 0.0

        self.model_name = model_name
        self.feature_encoder = feature_encoder
        self.relation_network = relation_network
        self.dhc_model = dhc_model
        self.compare_fsl_fn = compare_fsl_fn

        self.feature_encoder_dir = Tools.new_dir("../models/{}_feature_encoder_{}way_{}shot.pkl".format(
            self.model_name, self.class_num, self.sample_num_per_class))
        self.relation_network_dir = Tools.new_dir("../models/{}_relation_network_{}way_{}shot.pkl".format(
            self.model_name, self.class_num, self.sample_num_per_class))
        self.dhc_model_dir = Tools.new_dir("../models/{}_dhc_model_{}way_{}shot.pkl".format(
            self.model_name, self.class_num, self.sample_num_per_class))

        # model
        self.feature_encoder.apply(self._weights_init).cuda()
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=self.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, self.train_episode//3, gamma=0.5)
        self.relation_network.apply(self._weights_init).cuda()
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=self.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, self.train_episode//3, gamma=0.5)
        self.dhc_model.apply(self._weights_init).cuda()
        self.dhc_model_optim = torch.optim.Adam(self.dhc_model.parameters(), lr=self.learning_rate)
        self.dhc_model_scheduler = StepLR(self.dhc_model_optim, self.train_episode//3, gamma=0.5)

        # data
        (self.folders_train, self.folders_val, self.folders_test,
         self.dhc_train_name, self.dhc_val_name, self.dhc_test_name) = MiniImageNetTask.folders(data_root)
        self.dhc_train_loader = DHCMiniImageNet.get_data_loader(self.dhc_train_name, batch_size=32, shuffle=False)
        self.dhc_val_loader = DHCMiniImageNet.get_data_loader(self.dhc_val_name, batch_size=32, shuffle=False)
        self.dhc_test_loader = DHCMiniImageNet.get_data_loader(self.dhc_test_name, batch_size=32, shuffle=False)

        # DHC
        self.ratio = 3
        self.out_dim = 200
        self.produce_class = DHCProduceClass(n_sample=len(self.dhc_train_name), out_dim=self.out_dim, ratio=self.ratio)

        self.dhc_loss = DHCLoss().cuda()
        self.fsl_loss = nn.MSELoss().cuda()
        self.fsl_dhc_loss = nn.MSELoss().cuda()

        self.summary_dir = summary_dir
        self.is_summary = self.summary_dir is not None
        self.writer = None
        pass

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data = torch.ones(m.bias.data.size())
            pass
        pass

    def load_model(self):
        if os.path.exists(self.feature_encoder_dir):
            self.feature_encoder.load_state_dict(torch.load(self.feature_encoder_dir))
            Tools.print("load feature encoder success from {}".format(self.feature_encoder_dir))

        if os.path.exists(self.relation_network_dir):
            self.relation_network.load_state_dict(torch.load(self.relation_network_dir))
            Tools.print("load relation network success from {}".format(self.relation_network_dir))

        if os.path.exists(self.dhc_model_dir):
            self.dhc_model.load_state_dict(torch.load(self.dhc_model_dir))
            Tools.print("load dhc model success from {}".format(self.dhc_model_dir))
        pass

    def _feature_vision(self, other_sample_features,
                        other_batch_features, other_relation_features, episode, is_vision=False):
        if self.is_summary and (episode % 20000 == 0 or is_vision):

            if self.writer is None:
                self.writer = SummaryWriter(self.summary_dir)
                pass

            feature_root_name = "Train" if episode >= 0 else "Val"

            # 特征可视化
            for other_features, name in [[other_sample_features, "Sample"],
                                         [other_batch_features, "Batch"],
                                         [other_relation_features, "Relation"]]:
                for key in other_features:
                    if other_features[key].size(1) == 3:  # 原图
                        one_features = vutils.make_grid(other_features[key], normalize=True,
                                                        scale_each=True, nrow=self.batch_num_per_class)
                        self.writer.add_image('{}-{}-{}'.format(feature_root_name, name, key), one_features, episode)
                        pass
                    else:  # 特征
                        key_features = torch.split(other_features[key], split_size_or_sections=1, dim=1)
                        for index, feature_one in enumerate(key_features):
                            one_features = vutils.make_grid(feature_one, normalize=True,
                                                            scale_each=True, nrow=self.batch_num_per_class)
                            self.writer.add_image('{}-{}-{}/{}'.format(feature_root_name, name,
                                                                       key, index), one_features, episode)
                            pass
                        pass
                    pass
                pass

            # 参数可视化
            for name, param in self.feature_encoder.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), episode)
                pass
            for name, param in self.relation_network.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), episode)
                pass
            for name, param in self.dhc_model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), episode)
                pass

            pass
        pass

    def compare_fsl_1(self, samples, batches):
        # calculate features
        sample_features, other_sample_features = self.feature_encoder(samples.cuda(), self.is_summary)  # 5x64*19*19
        batch_features, other_batch_features = self.feature_encoder(batches.cuda(), self.is_summary)  # 75x64*19*19

        batch_size, feature_dim, feature_width, feature_height = batch_features.shape

        # calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(
            self.sample_num_per_class * self.class_num, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext),
                                   2).view(-1, feature_dim * 2, feature_width, feature_height)

        relations, other_relation_features = self.relation_network(relation_pairs, self.is_summary)
        relations = relations.view(-1, self.class_num * self.sample_num_per_class)

        return (relations, sample_features, batch_features,
                other_sample_features, other_batch_features, other_relation_features)

    def train(self):
        Tools.print()
        Tools.print("Training...")

        if self.is_summary and self.writer is None:
            self.writer = SummaryWriter(self.summary_dir)
            pass

        all_loss, all_loss_fsl, all_loss_fsl_dhc = 0.0, 0.0, 0.0
        all_loss_dhc_sample, all_loss_dhc_batch, all_loss_dhc_batch_2 = 0.0, 0.0, 0.0
        for episode in range(self.train_episode):
            ###########################################################################
            # 0 更新标签
            if episode % (self.val_freq * 1) == 0:
                Tools.print()
                Tools.print("Update label {} .......".format(episode))
                self.feature_encoder.eval()
                self.dhc_model.eval()

                self.produce_class.reset()
                for batch_idx, (inputs, labels, indexes) in enumerate(self.dhc_train_loader):
                    sample_features, _ = self.feature_encoder(inputs.cuda())  # 5x64*19*19
                    _, out_l2norm, _ = self.dhc_model(sample_features)
                    self.produce_class.cal_label(out_l2norm, indexes.cuda())
                    pass

                Tools.print("Epoch: [{}] 1-{}/{}".format(episode, self.produce_class.count, self.produce_class.count_2))
                pass

            # test dhc
            if episode % (self.val_freq * 1) == 0:
                Tools.print()
                Tools.print("Test DHC {} .......".format(episode))
                self.feature_encoder.eval()
                self.dhc_model.eval()

                _acc_1, _acc_2 = KNN.knn(episode, self.feature_encoder,
                                         self.dhc_model, self.out_dim, self.dhc_train_loader, 100)
                Tools.print("Epoch: [{}] Train {:.4f}/{:.4f}".format(episode, _acc_1, _acc_2))

                _acc_1, _acc_2 = KNN.knn(episode, self.feature_encoder,
                                         self.dhc_model, self.out_dim, self.dhc_val_loader, 100)
                Tools.print("Epoch: [{}] Val {:.4f}/{:.4f}".format(episode, _acc_1, _acc_2))

                _acc_1, _acc_2 = KNN.knn(episode, self.feature_encoder,
                                         self.dhc_model, self.out_dim, self.dhc_test_loader, 100)
                Tools.print("Epoch: [{}] Test {:.4f}/{:.4f}".format(episode, _acc_1, _acc_2))

                Tools.print()
                pass

            # val fsl
            if episode % self.val_freq == 0:
                Tools.print()
                Tools.print("Valing...")
                train_accuracy = self.val_train(episode, is_print=True)
                val_accuracy = self.val(episode, is_print=True)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.feature_encoder.state_dict(), self.feature_encoder_dir)
                    torch.save(self.relation_network.state_dict(), self.relation_network_dir)
                    torch.save(self.dhc_model.state_dict(), self.dhc_model_dir)
                    Tools.print("Save networks for episode: {}".format(episode))
                    pass

                if self.is_summary:
                    self.writer.add_scalar('loss/avg-loss', all_loss / (episode % self.val_freq), episode)
                    self.writer.add_scalar('accuracy/val', val_accuracy, episode)
                    self.writer.add_scalar('accuracy/train', train_accuracy, episode)
                    pass

                all_loss, all_loss_fsl, all_loss_fsl_dhc = 0.0, 0.0, 0.0
                all_loss_dhc_sample, all_loss_dhc_batch, all_loss_dhc_batch_2 = 0.0, 0.0, 0.0
                Tools.print()
                pass
            ###########################################################################

            self.feature_encoder.train()
            self.dhc_model.train()

            ###########################################################################
            # 1 init dataset
            task = MiniImageNetTask(self.folders_train, self.class_num,
                                    self.sample_num_per_class, self.batch_num_per_class, self.dhc_train_name)
            sample_data_loader = MiniImageNet.get_data_loader(task, self.sample_num_per_class, "train", shuffle=False)
            batch_data_loader = MiniImageNet.get_data_loader(task, self.batch_num_per_class, split="val", shuffle=True)
            samples, sample_labels, dhc_sample_ids = sample_data_loader.__iter__().next()
            batches, batch_labels, dhc_batch_ids = batch_data_loader.__iter__().next()
            ###########################################################################

            ###########################################################################
            # 2 calculate features
            (relations, sample_features, batch_features, other_sample_features,
             other_batch_features, other_relation_features) = self.compare_fsl_fn(self, samples, batches)
            dhc_sample_out_logits, _, dhc_sample_out_sigmoid = self.dhc_model(sample_features)
            dhc_batch_out_logits, _, dhc_batch_out_sigmoid = self.dhc_model(batch_features)

            # 可视化
            self._feature_vision(other_sample_features, other_batch_features, other_relation_features, episode)
            ###########################################################################

            ###########################################################################
            # 3 loss
            dhc_sample_targets = self.produce_class.get_label(dhc_sample_ids.cuda())
            dhc_batch_targets = self.produce_class.get_label(dhc_batch_ids.cuda())
            dhc_batch_targets_2 = torch.index_select(dhc_sample_targets, 0, batch_labels.cuda())
            dhc_sample_mse_sigmoid = torch.index_select(dhc_sample_out_sigmoid, 0, batch_labels.cuda())

            one_hot_labels = torch.zeros(self.batch_num_per_class * self.class_num,
                                         self.class_num).scatter_(1, batch_labels.view(-1, 1), 1).cuda()

            loss_fsl = self.fsl_loss(relations, one_hot_labels) * 1.0  # a
            loss_fsl_dhc = self.fsl_dhc_loss(dhc_batch_out_sigmoid, dhc_sample_mse_sigmoid) * 1.0

            loss_dhc_sample = self.dhc_loss(dhc_sample_out_logits, dhc_sample_targets) * 0.04
            loss_dhc_batch = self.dhc_loss(dhc_batch_out_logits, dhc_batch_targets) * 0.04
            loss_dhc_batch_2 = self.dhc_loss(dhc_batch_out_logits, dhc_batch_targets_2) * 0.0  # * 0.02

            loss = loss_fsl + loss_dhc_sample + loss_dhc_batch + loss_dhc_batch_2 + loss_fsl_dhc

            all_loss += loss.item()
            all_loss_fsl += loss_fsl.item()
            all_loss_dhc_sample += loss_dhc_sample.item()
            all_loss_dhc_batch += loss_dhc_batch.item()
            all_loss_dhc_batch_2 += loss_dhc_batch_2.item()
            all_loss_fsl_dhc += loss_fsl_dhc.item()
            ###########################################################################

            ###########################################################################
            # 4 backward
            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()
            self.dhc_model.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
            self.feature_encoder_optim.step()
            self.feature_encoder_scheduler.step(episode)

            torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)
            self.relation_network_optim.step()
            self.relation_network_scheduler.step(episode)

            torch.nn.utils.clip_grad_norm_(self.dhc_model.parameters(), 0.5)
            self.dhc_model_optim.step()
            self.dhc_model_scheduler.step(episode)
            ###########################################################################

            ###########################################################################
            # 5 summary print
            if self.is_summary:
                self.writer.add_scalar('loss/loss-total', loss.item(), episode)
                self.writer.add_scalar('loss/loss-fsl', loss_fsl.item(), episode)
                self.writer.add_scalar('loss/loss-dhc-sample', loss_dhc_sample.item(), episode)
                self.writer.add_scalar('loss/loss-dhc-batch', loss_dhc_batch.item(), episode)
                self.writer.add_scalar('loss/loss-fsl-dhc', loss_fsl_dhc.item(), episode)
                self.writer.add_scalar('learning-rate', self.feature_encoder_scheduler.get_lr(), episode)
                pass

            # print
            if (episode + 1) % self.print_freq == 0:
                Tools.print(
                    "{:6} loss:{:.3f}/{:.3f} fsl:{:.3f}/{:.3f} dhc-s:{:.3f}/{:.3f} "
                    "dhc-b:{:.3f}/{:.3f} dhc-b-2:{:.3f}/{:.3f} fsl-dhc:{:.3f}/{:.3f} lr:{}".format(
                        episode + 1,
                        all_loss / (episode % self.val_freq + 1), loss.item(),
                        all_loss_fsl / (episode % self.val_freq + 1), loss_fsl.item(),
                        all_loss_dhc_sample / (episode % self.val_freq + 1), loss_dhc_sample.item(),
                        all_loss_dhc_batch / (episode % self.val_freq + 1), loss_dhc_batch.item(),
                        all_loss_dhc_batch_2 / (episode % self.val_freq + 1), loss_dhc_batch_2.item(),
                        all_loss_fsl_dhc / (episode % self.val_freq + 1), loss_fsl_dhc.item(),
                        self.feature_encoder_scheduler.get_lr()))
                pass
            ###########################################################################
            pass

        pass

    def _val(self, folders, sampler_test, all_episode, episode=-1):
        accuracies = []
        for i in range(all_episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = MiniImageNetTask(folders, self.class_num, self.sample_num_per_class, self.batch_num_per_class)
            sample_data_loader = MiniImageNet.get_data_loader(task, 1, "train", sampler_test, shuffle=False)
            batch_data_loader = MiniImageNet.get_data_loader(task, 3, "val",  sampler_test=sampler_test, shuffle=True)
            samples, labels, _ = sample_data_loader.__iter__().next()

            for batches, batch_labels, _ in batch_data_loader:
                ###########################################################################
                # calculate features
                (relations, sample_features, batch_features, other_sample_features,
                 other_batch_features, other_relation_features) = self.compare_fsl_fn(self, samples, batches)

                # 可视化
                self._feature_vision(other_sample_features, other_batch_features, other_relation_features,
                                     episode if episode >= 0 else -episode, is_vision=False if episode >= 0 else True)
                ###########################################################################

                _, predict_labels = torch.max(relations.data, 1)
                batch_size = batch_labels.shape[0]
                rewards = [1 if predict_labels[j].cpu() == batch_labels[j] else 0 for j in range(batch_size)]
                total_rewards += np.sum(rewards)

                counter += batch_size
                pass

            accuracies.append(total_rewards / 1.0 / counter)
            pass
        return np.mean(np.array(accuracies, dtype=np.float))

    def val_train(self, episode, is_print=True):
        val_train_accuracy = self._val(self.folders_train, sampler_test=False,
                                       all_episode=self.val_episode, episode=episode)
        if is_print:
            Tools.print("Val Train Accuracy: {}".format(val_train_accuracy))
            pass
        return val_train_accuracy

    def val(self, episode, is_print=True):
        val_accuracy = self._val(self.folders_val, sampler_test=False, all_episode=self.val_episode, episode=episode)
        if is_print:
            Tools.print("Val Accuracy: {}".format(val_accuracy))
            pass
        return val_accuracy

    def test(self):
        Tools.print()
        Tools.print("Testing...")
        total_accuracy = 0.0
        for episode in range(self.test_avg_num):
            test_accuracy = self._val(self.folders_test, sampler_test=True, all_episode=self.test_episode)
            total_accuracy += test_accuracy
            Tools.print("episode={}, Test accuracy={}, Total accuracy={}".format(
                episode, test_accuracy, total_accuracy))
            pass

        final_accuracy = total_accuracy / self.test_avg_num
        Tools.print("Final accuracy: {}".format(final_accuracy))
        return final_accuracy

    pass


##############################################################################################################


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 数据增强，大的无监督网络，

    # 0.7015 / 0.5001
    # 0.7038 / 0.5209
    # 0.6790 / 0.5189
    # 0.6919 / 0.5198  # small net, fsl+dhc-s+dhc-b+fsl-dhc(cross)
    # 0.7100 / 0.5278  # large net, fsl+dhc-s+dhc-b
    _model_name = "8"
    _feature_encoder = CNNEncoder()
    _relation_network = RelationNetwork()
    _dhc_model = DHCModel(in_dim=64, out_dim=200)
    _compare_fsl_fn = Runner.compare_fsl_1

    Tools.print("{}".format(_model_name))
    # _summary_dir = Tools.new_dir("../log/{}".format(_model_name))
    _summary_dir = None

    runner = Runner(model_name=_model_name, feature_encoder=_feature_encoder, relation_network=_relation_network,
                    dhc_model=_dhc_model, compare_fsl_fn=_compare_fsl_fn, train_episode=300000,
                    data_root='/mnt/4T/Data/miniImagenet', summary_dir=_summary_dir)

    runner.load_model()

    # runner.val(episode=0)
    # runner.val_train(episode=0)
    # runner.test()

    runner.train()

    runner.load_model()
    runner.val_train(episode=runner.train_episode)
    runner.val(episode=runner.train_episode)
    runner.test()
