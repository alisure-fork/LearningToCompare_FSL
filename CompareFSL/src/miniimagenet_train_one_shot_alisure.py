import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset


##############################################################################################################


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

    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = dict(zip(class_folders, np.array(range(len(class_folders)))))

        samples = dict()
        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]
            pass

        self.train_labels = [labels[os.path.split(x)[0]] for x in self.train_roots]
        self.test_labels = [labels[os.path.split(x)[0]] for x in self.test_roots]
        pass

    pass


class MiniImageNet(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None, data_dict=None):
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.data_dict = data_dict
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        pass

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        if self.data_dict:
            image = self.data_dict[image_root]
        else:
            image = Image.open(image_root)
            image = image.convert('RGB')
            pass
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

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
        return folders_train, folders_val, folders_test

    @staticmethod
    def load_data(folders_train, folders_val, folders_test):
        data_dict = {}
        Tools.print()
        Tools.print("Begin to load data")
        for folders in [folders_train, folders_val, folders_test]:
            for folder in folders:
                image_paths = [os.path.join(folder, x) for x in os.listdir(folder)]
                for image_path in image_paths:
                    image = Image.open(image_path)
                    image = image.convert('RGB')
                    data_dict[image_path] = image
                    pass
                pass
            pass
        Tools.print("End to load data")
        Tools.print()
        return data_dict

    @staticmethod
    def get_data_loader(task, num_per_class=1, split='train', sampler_test=False, shuffle=False, data_dict=None):
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        transform_train = transforms.Compose([transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)), transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #     transforms.RandomGrayscale(p=0.2), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        # transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        if split == 'train':
            dataset = MiniImageNet(task, split=split, data_dict=data_dict, transform=transform_train)
            sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
        else:
            dataset = MiniImageNet(task, split=split, data_dict=data_dict, transform=transform_test)
            if not sampler_test:
                sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
            else:  # test
                sampler = ClassBalancedSamplerTest(task.num_classes, task.test_num, shuffle=shuffle)
                pass
            pass

        return DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    pass


##############################################################################################################


# Original 1
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


##############################################################################################################


class Runner(object):

    def __init__(self, model_name, feature_encoder, relation_network, compare_fsl_fn, train_episode=300000,
                 data_root='/mnt/4T/Data/miniImagenet', summary_dir=None, is_load_data=False):
        self.class_num = 5
        self.sample_num_per_class = 1
        self.batch_num_per_class = 15

        self.train_episode = train_episode  # 500000
        self.val_episode = 600
        self.test_avg_num = 10
        self.test_episode = 600

        self.learning_rate = 0.001

        self.print_freq = 1000
        self.val_freq = 5000  # 5000
        self.best_accuracy = 0.0

        self.model_name = model_name
        self.feature_encoder = feature_encoder
        self.relation_network = relation_network
        self.compare_fsl_fn = compare_fsl_fn

        self.feature_encoder_dir = Tools.new_dir("../models/{}_feature_encoder_{}way_{}shot.pkl".format(
            self.model_name, self.class_num, self.sample_num_per_class))
        self.relation_network_dir = Tools.new_dir("../models/{}_relation_network_{}way_{}shot.pkl".format(
            self.model_name, self.class_num, self.sample_num_per_class))

        # data
        self.folders_train, self.folders_val, self.folders_test = MiniImageNet.folders(data_root)
        self.data_dict = MiniImageNet.load_data(self.folders_train, self.folders_val,
                                                self.folders_test) if is_load_data else None

        # model
        self.feature_encoder.apply(self._weights_init).cuda()
        self.relation_network.apply(self._weights_init).cuda()
        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(), lr=self.learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim, self.train_episode//3, gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(), lr=self.learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim, self.train_episode//3, gamma=0.5)

        self.loss = self._loss()

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
            m.bias.data = torch.ones(m.bias.data.size())
            pass
        pass

    @staticmethod
    def _loss():
        mse = nn.MSELoss().cuda()
        return mse

    def load_model(self):
        if os.path.exists(self.feature_encoder_dir):
            self.feature_encoder.load_state_dict(torch.load(self.feature_encoder_dir))
            Tools.print("load feature encoder success from {}".format(self.feature_encoder_dir))

        if os.path.exists(self.relation_network_dir):
            self.relation_network.load_state_dict(torch.load(self.relation_network_dir))
            Tools.print("load relation network success from {}".format(self.relation_network_dir))
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

            pass
        pass

    # Original 1
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

        return relations, other_sample_features, other_batch_features, other_relation_features

    def train(self):
        Tools.print()
        Tools.print("Training...")

        if self.is_summary and self.writer is None:
            self.writer = SummaryWriter(self.summary_dir)
            pass

        all_loss = 0.0
        for episode in range(self.train_episode):
            # init dataset
            task = MiniImageNetTask(self.folders_train, self.class_num,
                                    self.sample_num_per_class, self.batch_num_per_class)
            sample_data_loader = MiniImageNet.get_data_loader(task, self.sample_num_per_class, "train",
                                                              shuffle=False, data_dict=self.data_dict)
            batch_data_loader = MiniImageNet.get_data_loader(task, self.batch_num_per_class, split="val",
                                                             shuffle=True, data_dict=self.data_dict)
            samples, sample_labels = sample_data_loader.__iter__().next()
            batches, batch_labels = batch_data_loader.__iter__().next()

            ###########################################################################
            # calculate features
            relations, other_sample_features, other_batch_features, other_relation_features = self.compare_fsl_fn(
                self, samples, batches)

            # 可视化
            self._feature_vision(other_sample_features, other_batch_features, other_relation_features, episode)
            ###########################################################################

            one_hot_labels = torch.zeros(self.batch_num_per_class * self.class_num,
                                         self.class_num).scatter_(1, batch_labels.view(-1, 1), 1).cuda()
            loss = self.loss(relations, one_hot_labels)

            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)

            self.feature_encoder_optim.step()
            self.relation_network_optim.step()
            self.feature_encoder_scheduler.step(episode)
            self.relation_network_scheduler.step(episode)

            if self.is_summary:
                self.writer.add_scalar('loss/now-loss', loss.item(), episode)
                self.writer.add_scalar('learning-rate', self.feature_encoder_scheduler.get_lr(), episode)
                pass

            all_loss += loss.item()
            if (episode + 1) % self.print_freq == 0:
                Tools.print("Episode: {} avg loss: {} loss: {} lr: {}".format(
                    episode + 1, all_loss / (episode % self.val_freq), loss.item(),
                    self.feature_encoder_scheduler.get_lr()))
                pass

            if (episode + 1) % self.val_freq == 0:
                Tools.print()
                Tools.print("Valing...")
                train_accuracy = runner.val_train(episode, is_print=True)
                val_accuracy = self.val(episode, is_print=True)

                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    torch.save(self.feature_encoder.state_dict(), self.feature_encoder_dir)
                    torch.save(self.relation_network.state_dict(), self.relation_network_dir)
                    Tools.print("Save networks for episode: {}".format(episode))
                    pass

                if self.is_summary:
                    self.writer.add_scalar('loss/avg-loss', all_loss / (episode % self.val_freq), episode)
                    self.writer.add_scalar('accuracy/val', val_accuracy, episode)
                    self.writer.add_scalar('accuracy/train', train_accuracy, episode)
                    pass

                all_loss = 0.0
                Tools.print()
                pass

            pass

        pass

    def _val(self, folders, sampler_test, all_episode, episode=-1):
        accuracies = []
        for i in range(all_episode):
            total_rewards = 0
            counter = 0
            # 随机选5类，每类中取出1个作为训练样本，每类取出15个作为测试样本
            task = MiniImageNetTask(folders, self.class_num, self.sample_num_per_class, self.batch_num_per_class)
            sample_data_loader = MiniImageNet.get_data_loader(task, 1, "train", sampler_test=sampler_test,
                                                              shuffle=False, data_dict=self.data_dict)
            batch_data_loader = MiniImageNet.get_data_loader(task, 3, "val", sampler_test=sampler_test,
                                                             shuffle=True, data_dict=self.data_dict)
            samples, labels = sample_data_loader.__iter__().next()

            for batches, batch_labels in batch_data_loader:
                ###########################################################################
                # calculate features
                relations, other_sample_features, other_batch_features, other_relation_features = self.compare_fsl_fn(
                    self, samples, batches)
                # 可视化
                self._feature_vision(other_sample_features, other_batch_features, other_relation_features,
                                     episode if episode > 0 else -episode, is_vision=False if episode > 0 else True)
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
    """
    tensorboard --logdir _model_name
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    _model_name = "1"
    if _model_name == "1":
        # 0.7015 / 0.50013
        _feature_encoder = CNNEncoder()
        _relation_network = RelationNetwork()
        _compare_fsl_fn = Runner.compare_fsl_1
        _model_name = "1"
    else:
        raise Exception("...............")

    Tools.print("{}".format(_model_name))
    # _summary_dir = Tools.new_dir("../log/{}".format(_model_name))
    _summary_dir = None

    runner = Runner(model_name=_model_name, feature_encoder=_feature_encoder, relation_network=_relation_network,
                    compare_fsl_fn=_compare_fsl_fn, train_episode=300000, is_load_data=False,
                    data_root='/mnt/4T/Data/miniImagenet', summary_dir=_summary_dir)

    runner.load_model()

    # runner.test()
    # runner.train()
    runner.val_train(episode=runner.train_episode)
    runner.val(episode=runner.train_episode)
    runner.test()
