import os
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy import io
import scipy.io as sio
from sklearn.decomposition import PCA
import random
import torchvision.transforms.functional as F


def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def load_data(data_set_name,
              data_path1=r''):#data path
    if data_set_name == 'Muufl':
        data1 = io.loadmat(os.path.join(data_path1, data_set_name, 'HSI.mat'))['HSI']
        data2 = io.loadmat(os.path.join(data_path1, data_set_name, 'LiDAR.mat'))['LiDAR']
        labels = io.loadmat(os.path.join(data_path1, data_set_name, 'gt.mat'))['gt']
        train = io.loadmat(os.path.join(data_path1, data_set_name, 'train_labels'))['train_labels']
        test = io.loadmat(os.path.join(data_path1, data_set_name, 'test_labels'))['test_labels']
    if data_set_name == 'Trento':
        data1 = io.loadmat(os.path.join(data_path1, data_set_name, 'HSI.mat'))['HSI']
        data2 = io.loadmat(os.path.join(data_path1, data_set_name, 'LiDAR.mat'))['LiDAR']
        labels = io.loadmat(os.path.join(data_path1, data_set_name, 'gt.mat'))['gt']
        train = io.loadmat(os.path.join(data_path1, data_set_name, 'train_labels'))['train_labels']
        test = io.loadmat(os.path.join(data_path1, data_set_name, 'test_labels'))['test_labels']
    if data_set_name == 'Houston':
        data1 = io.loadmat(os.path.join(data_path1, data_set_name, 'HSI.mat'))['HSI']
        data2 = io.loadmat(os.path.join(data_path1, data_set_name, 'LiDAR.mat'))['LiDAR']
        labels = io.loadmat(os.path.join(data_path1, data_set_name, 'gt.mat'))['gt']
        train = io.loadmat(os.path.join(data_path1, data_set_name, 'train_labels'))['train_labels']
        test = io.loadmat(os.path.join(data_path1, data_set_name, 'test_labels'))['test_labels']
    if data_set_name == 'Augsburg':
        data1 = io.loadmat(os.path.join(data_path1, data_set_name, 'HSI.mat'))['HSI']
        data2 = io.loadmat(os.path.join(data_path1, data_set_name, 'LiDAR.mat'))['LiDAR']
        labels = io.loadmat(os.path.join(data_path1, data_set_name, 'gt.mat'))['gt']
        train = io.loadmat(os.path.join(data_path1, data_set_name, 'train_labels'))['train_labels']
        test = io.loadmat(os.path.join(data_path1, data_set_name, 'test_labels'))['test_labels']
    if data_set_name == 'Berlin':
        data1 = io.loadmat(os.path.join(data_path1, data_set_name, 'HSI.mat'))['HSI']
        data2 = io.loadmat(os.path.join(data_path1, data_set_name, 'LiDAR.mat'))['LiDAR']
        labels = io.loadmat(os.path.join(data_path1, data_set_name, 'gt.mat'))['gt']
        train = io.loadmat(os.path.join(data_path1, data_set_name, 'train_labels'))['train_labels']
        test = io.loadmat(os.path.join(data_path1, data_set_name, 'test_labels'))['test_labels']

    return data1, data2, labels, train, test

def apply_pca(x, num_components):

    y = np.reshape(x, (-1, x.shape[2]))
    pca = PCA(n_components=num_components, whiten=True) # whiten=true
    y = pca.fit_transform(y) # pca
    y = np.reshape(y, (x.shape[0], x.shape[1], num_components))
    #print("yes")
    return y


def loadtrandte_data(data_set_name):
    if data_set_name == 'Muufl':
        data1 = io.loadmat(os.path.join(data_set_name, 'train_labels'))['train_labels']
        data2 = io.loadmat(os.path.join( data_set_name, 'test_labels'))['test_labels']
    if data_set_name == 'Trento':
        data1 = io.loadmat(os.path.join(data_set_name, 'train_labels'))['train_labels']
        data2 = io.loadmat(os.path.join( data_set_name, 'test_labels'))['test_labels']
    if data_set_name == 'Houston':
        data1 = io.loadmat(os.path.join(data_set_name, 'train_labels'))['train_labels']
        data2 = io.loadmat(os.path.join( data_set_name, 'test_labels'))['test_labels']
    if data_set_name == 'Berlin':
        data1 = io.loadmat(os.path.join(data_set_name, 'train_labels'))['train_labels']
        data2 = io.loadmat(os.path.join( data_set_name, 'test_labels'))['test_labels']

    return data1, data2

def normalize(input2):
    input2_normalize = np.zeros(input2.shape, dtype=np.float32)

    if len(input2.shape) == 2:  # (height, width)
        input2_max = np.max(input2)
        input2_min = np.min(input2)
        input2_normalize = (input2 - input2_min) / (input2_max - input2_min)
    elif len(input2.shape) == 3:  #(height, width, bands)
        for i in range(input2.shape[2]):
            input2_max = np.max(input2[:, :, i])
            input2_min = np.min(input2[:, :, i])

            input2_normalize[:, :, i] = (input2[:, :, i] - input2_min) / (input2_max - input2_min)
    else:
        raise ValueError("Input data must be 2D (height, width) or 3D (height, width, bands).")

    return input2_normalize



def ImageStretching(image):
    channels = image.shape[2]
    band_list = []
    for i in range(channels):
        band_data = image[:,:,i]
        band_min = np.percentile(band_data,2)
        band_max = np.percentile(band_data,98)
        band_data = (band_data - band_min) / (band_max - band_min)
        band_list.append(band_data)
    image_data = np.stack(band_list, axis=-1)
    image_data = np.clip(image_data, 0, 1)
    image_data = (image_data * 255).astype(np.uint8)
    image_data = np.uint8(image_data)

    return image_data

def padpatch(Data1, Data2, Data3, patchsize, pad_width):
    offset = patchsize % 2
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)

    Data3 = Data3.reshape([m1, n1, -1])
    [m3, n3, l3] = np.shape(Data3)

    x1 = Data1
    x2 = Data2
    x3 = Data3

    x1_pad = np.empty((m1 + patchsize+offset, n1 + patchsize+offset, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize+offset, n2 + patchsize+offset, l2), dtype='float32')
    x3_pad = np.empty((m3 + patchsize+offset, n3 + patchsize+offset, l3), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x2_pad[:, :, i] = temp2
    for i in range(l3):
        temp = x3[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x3_pad[:, :, i] = temp2

    return x1_pad,x2_pad,x3_pad


def sampling(ratio_list, num_list, gt_reshape, class_count, Flag):
    all_label_index_dict, train_label_index_dict, val_label_index_dict, test_label_index_dict = {}, {}, {}, {}
    all_label_index_list, train_label_index_list, val_label_index_list, test_label_index_list = [], [], [], []
    for cls in range(class_count):
        cls_index = np.where(gt_reshape == cls + 1)[0]
        all_label_index_dict[cls] = list(cls_index)
        np.random.shuffle(cls_index)
        if Flag == 0:  # Fixed proportion for each category
            train_index_flag = max(int(ratio_list[0] * len(cls_index)), 3)  # at least 3 samples per class]
            val_index_flag = max(int(ratio_list[1] * len(cls_index)), 1)
        # Split by num per class
        elif Flag == 1:  # Fixed quantity per category
            if len(cls_index) > num_list[0]:
                train_index_flag = num_list[0]
            else:
                train_index_flag = 15
            val_index_flag = num_list[1]
        train_label_index_dict[cls] = list(cls_index[:train_index_flag])
        test_label_index_dict[cls] = list(cls_index[train_index_flag:][val_index_flag:])
        val_label_index_dict[cls] = list(cls_index[train_index_flag:][:val_index_flag])
        train_label_index_list += train_label_index_dict[cls]
        test_label_index_list += test_label_index_dict[cls]
        val_label_index_list += val_label_index_dict[cls]
        all_label_index_list += all_label_index_dict[cls]

    return train_label_index_list, val_label_index_list, test_label_index_list, all_label_index_list

def index_assignment(index, row, col, pad_length):
    new_assign = {}  # dictionary.
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]

    return new_assign

def split_train_test_labels(label_matrix, num_samples_per_class, dataset, random_seed=None):
    np.random.seed(random_seed)
    num_classes = label_matrix.max() # 11
    train_indices = []
    test_indices = []
    for class_label in range(1, num_classes + 1):
        class_indices = np.where(label_matrix == class_label)
        class_indices = np.array(class_indices).T
        # Shuffle indices
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:num_samples_per_class])
        test_indices.extend(class_indices[num_samples_per_class:])
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    train_labels = np.zeros_like(label_matrix)
    test_labels = np.copy(label_matrix)
    # Construct training set
    for idx in train_indices:
        train_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]
    # Construct test set
    for idx in test_indices:
        test_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]
    # Set selected training indices to 0 in test set
    test_labels[train_indices[:, 0], train_indices[:, 1]] = 0
    print(f"Number of training samples: {len(train_indices)}")
    print(f"Number of test samples: {len(test_indices)}")
    dataset_folder = f"./{dataset}"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    train_file_path = os.path.join(dataset_folder, "train_labels.mat")
    test_file_path = os.path.join(dataset_folder, "test_labels.mat")
    sio.savemat(train_file_path, {'train_labels': train_labels})
    sio.savemat(test_file_path, {'test_labels': test_labels}) #

    return train_labels, test_labels


def list_split_train_test_labels(label_matrix, num_samples_per_class_list, dataset, random_seed=None):
    np.random.seed(random_seed)
    num_classes = label_matrix.max()


    if len(num_samples_per_class_list) != num_classes:
        raise ValueError(
            f"num_samples_per_class_list length({len(num_samples_per_class_list)})should be ({num_classes})")

    train_indices = []
    test_indices = []

    for class_label in range(1, num_classes + 1):
        class_indices = np.where(label_matrix == class_label)
        class_indices = np.array(class_indices).T

        num_samples = num_samples_per_class_list[class_label - 1]

        if num_samples > len(class_indices):
            print(
                f"warning: class {class_label} number of samples ({num_samples})over the true number ({len(class_indices)})")
            num_samples = len(class_indices)


        np.random.shuffle(class_indices)


        train_indices.extend(class_indices[:num_samples])
        test_indices.extend(class_indices[num_samples:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)


    train_labels = np.zeros_like(label_matrix)
    test_labels = np.copy(label_matrix)

    for idx in train_indices:
        train_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]

    for idx in test_indices:
        test_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]


    test_labels[train_indices[:, 0], train_indices[:, 1]] = 0

    print(f"train: {len(train_indices)}")
    print(f"test: {len(test_indices)}")


    for class_label in range(1, num_classes + 1):
        train_count = np.sum(train_labels == class_label)
        test_count = np.sum(test_labels == class_label)
        print(f"class {class_label}: train {train_count}, test {test_count}")


    dataset_folder = f"./{dataset}"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    train_file_path = os.path.join(dataset_folder, "train_labels.mat")
    test_file_path = os.path.join(dataset_folder, "test_labels.mat")
    sio.savemat(train_file_path, {'train_labels': train_labels})
    sio.savemat(test_file_path, {'test_labels': test_labels})

    return train_labels, test_labels

def trainone_patch(Data1,patchsize, pad_width, Label, ALL_Indices):
    offset = patchsize % 2
    [m1, n1, l1] = np.shape(Data1)
    x1 = Data1
    x1_pad = np.empty((m1 + patchsize+offset, n1 + patchsize+offset, l1), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x1_pad[:, :, i] = temp2
    # construct the training and testing set
    if ALL_Indices:
        [ind1, ind2] = np.where(Label >= 0)
    else:
        [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width+offset), (ind4[i] - pad_width):(ind4[i] + pad_width+offset), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()

    return TrainPatch1, TrainLabel

def traintwo_patch(Data1, Data2, patchsize, pad_width, Label, ALL_Indices,apply_augmentation=False):
    offset = patchsize % 2 #
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)
    x1 = Data1
    x2 = Data2
    x1_pad = np.empty((m1 + patchsize+offset, n1 + patchsize+offset, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize+offset, n2 + patchsize+offset, l2), dtype='float32')

    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x2_pad[:, :, i] = temp2
    # construct the training and testing set
    if ALL_Indices:
        [ind1, ind2] = np.where(Label >= 0)
    else:
        [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width+offset), (ind4[i] - pad_width):(ind4[i] + pad_width+offset), :]
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width + offset),
                 (ind4[i] - pad_width):(ind4[i] + pad_width + offset), :]


        patch1 = np.transpose(patch1, (2, 0, 1))
        patch2 = np.transpose(patch2, (2, 0, 1))


        if apply_augmentation:

            flip_type = random.choice(['none', 'horizontal'])

            if flip_type == 'horizontal':

                patch1 = np.flip(patch1, axis=2)
                patch2 = np.flip(patch2, axis=2)


        TrainPatch1[i, :, :, :] = patch1
        TrainPatch2[i, :, :, :] = patch2
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()

    return TrainPatch1, TrainPatch2, TrainLabel

def trainthird_patch(Data1, Data2, Data3, patchsize, pad_width, Label, ALL_Indices):
    offset = patchsize % 2
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)



    Data3 = Data3.reshape([m1, n1, -1])
    [m3, n3, l3] = np.shape(Data3)

    x1 = Data1
    x2 = Data2
    x3 = Data3
    x1_pad = np.empty((m1 + patchsize+offset, n1 + patchsize+offset, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize+offset, n2 + patchsize+offset, l2), dtype='float32')
    x3_pad = np.empty((m3 + patchsize+offset, n3 + patchsize+offset, l3), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x2_pad[:, :, i] = temp2
    for i in range(l3):
        temp = x3[:, :, i]
        temp2 = np.pad(temp, pad_width+offset, 'symmetric')
        x3_pad[:, :, i] = temp2
    # construct the training and testing set
    if ALL_Indices:
        [ind1, ind2] = np.where(Label >= 0)
    else:
        [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainPatch3 = np.empty((TrainNum, l3, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width+offset), (ind4[i] - pad_width):(ind4[i] + pad_width+offset), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width+offset), (ind4[i] - pad_width):(ind4[i] + pad_width+offset), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patch3 = x3_pad[(ind3[i] - pad_width):(ind3[i] + pad_width+offset), (ind4[i] - pad_width):(ind4[i] + pad_width+offset), :]
        patch3 = np.transpose(patch3, (2, 0, 1))
        TrainPatch3[i, :, :, :] = patch3
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainPatch3 = torch.from_numpy(TrainPatch3)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()

    return TrainPatch1, TrainPatch2, TrainPatch3, TrainLabel







