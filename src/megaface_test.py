import os 
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


class MegaDataset(Dataset):
    def __init__(self, path_list, resize_hw):
        self.images = []
        self.labels = []
        self.resize = (resize_hw[1], resize_hw[0])

        for line in path_list:
            line = line.strip()
            self.images.append(line)
            self.labels.append(line.split('/')[-2])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.images[idx])
        img_bgr = cv2.resize(img_bgr, self.resize)

        img_input = ((img_bgr / 255.) - 0.5) / 0.5
        img_input = torch.tensor(img_input.transpose(2, 0, 1)).float()

        label = self.labels[idx]

        return img_input, label

def get_mega_dataloader(root_dir, folder, batch_size, resize_hw=(112, 112)):
    path_list = []
    for name in os.listdir(os.path.join(root_dir, folder)):
        for img_name in os.listdir(os.path.join(root_dir, folder, name)):
            img_path = os.path.join(root_dir, folder, name, img_name)
            if img_name[-3:] in ['jpg', 'png']:
                path_list.append(img_path)
            elif '.' not in img_name:
                for img_name2 in os.listdir(img_path):
                    path_list.append(os.path.join(img_path, img_name2))
    dataset = MegaDataset(path_list, resize_hw)
    dataloader = DataLoader(dataset, batch_size, num_workers=8)
    return dataloader


@torch.no_grad()
def get_acc(model, mega_dataloaders, device='none'):
    feature_dict_arr = [{} for _ in range(len(mega_dataloaders))] 
    for i in range(len(mega_dataloaders)):
        for inputs, labels in tqdm(mega_dataloaders[i], f'[{device}] extract features {i}'):
            features = model(inputs.cuda()).detach().cpu().numpy()

            for feature, label in zip(features, labels):
                features_arr = feature_dict_arr[i].get(label, [])
                features_arr.append(feature)
                feature_dict_arr[i][label] = features_arr

    distractor_features = []
    for key, val in feature_dict_arr[-1].items():
        distractor_features.extend(val)
    distractor_features = np.array(distractor_features)
    
    acc_arr = []
    for i in range(len(feature_dict_arr[:-1])):
        query_dict = feature_dict_arr[i]
        count = [0, 0]
        with tqdm(query_dict.items(), f'[{device}] acc') as t:
            for key, val in t:
                query_features = np.array(val)
                qq_sim = np.dot(query_features, query_features.T)

                qd_sim = np.dot(query_features, distractor_features.T)
                qd_sim_max = [max(m) for m in qd_sim]

                for i in range(len(query_features)):
                    for j in range(i + 1, len(query_features), 1):
                        if qq_sim[i][j] > qd_sim_max[i]: count[0] += 1
                        if qq_sim[i][j] > qd_sim_max[j]: count[0] += 1
                        count[1] += 2

                print(count)
                t.set_postfix(str=f'acc: {count[0]/count[1]*100:.2f}%')

        acc_arr.append(count[0]/count[1])

    return acc_arr


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='vit_s')
    parser.add_argument('--input_size', default=[112, 112])
    parser.add_argument('--patch_size', default=8)
    parser.add_argument('--freeze_patch_embed', default=True)
    parser.add_argument('--num_features', default=512)
    parser.add_argument('--ckpt', default='output/backbone_faces.pth')
    parser.add_argument('--megaface_data_root', default='/file/tian/data/megaface_clean')
    parser.add_argument('--megaface_face_folders', default='faces,facescrub_images,megaface_images')
    cfg = parser.parse_args()
    print(cfg)

    megaface_folders = cfg.megaface_face_folders.split(',')
    megaface_dataloaders = []
    for i in range(len(megaface_folders)):
        megaface_dataloader = get_mega_dataloader(cfg.megaface_data_root, 
                                                  megaface_folders[i], 
                                                  512, 
                                                  cfg.input_size)
        megaface_dataloaders.append(megaface_dataloader)

    from accelerate import Accelerator
    from backbones import get_backbone
    accelerator = Accelerator(mixed_precision='fp16') 
    backbone = get_backbone(cfg)

    model = accelerator.prepare(backbone)
    model.eval()
    with accelerator.autocast():
        acc_arr = get_acc(model, megaface_dataloaders)
    print(acc_arr)
