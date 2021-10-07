""" 추론 코드

TODO:

NOTES:

REFERENCE:
    * MNC 코드 템플릿 predict.py

UPDATED:
"""
import os
import random
import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules.dataset import TestDataset
from modules.trainer import Trainer
from modules.utils import load_yaml, save_csv
import torch
from models.models import *
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from modules.metrics import get_metric_fn
import torch.nn as nn
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# # CONFIG
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)

PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/predict_config.yml')
config = load_yaml(PREDICT_CONFIG_PATH)


DATA_DIR = config['DIRECTORY']['data']
# MODEL = config['MODEL']
# TRAINED_MODEL_PATH = "./results/train/" + config['DIRECTORY']['model'] + "/best.pt"


## for Ensemble
model_list, model_path_list = [], []
for model in config['MODEL'].split():
    model_list.append(model)
for model_path in config['DIRECTORY']['model'].split():
    model_path_list.append("./results/train/done/" + model_path + "/best.pt")
# print(model_list, model_path_list)


# SEED
RANDOM_SEED = config['SEED']['random_seed']

# PREDICT
BATCH_SIZE = config['PREDICT']['batch_size']
INPUT_SHAPE = (224, 224)


if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    # SAVE_PATH = config['DIRECTORY']['model'] + '/pred.csv'  # for single model
    SAVE_PATH = '/multi_pred_final.csv'  # for multi model ensemble

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TestDataset(data_dir=DATA_DIR, input_shape=INPUT_SHAPE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8,
        pin_memory=True,)
    print('Test set samples:',len(test_dataset))


    # # # Load Single Model
    MODEL = model_list[0]
    TRAINED_MODEL_PATH = model_path_list[0]
    model = get_my_model(model_name=MODEL, num_classes = 10, checkpoint_path="").to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])

    pred_lst = []
    file_name_lst = []

    with torch.no_grad():
        for img, file_name in tqdm(test_dataloader):
            img = img.to(device)
            pred = model(img)
            pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
            file_name_lst.extend(file_name)
    df = pd.DataFrame({'file_name':file_name_lst, 'answer':pred_lst})
    df.to_csv(SAVE_PATH, index=None)


    # # # Load Multi Model -> Voting with argmax results
    # total_pred = []
    # for idx in range(len(model_list)):
    #     MODEL = model_list[idx]
    #     TRAINED_MODEL_PATH = model_path_list[idx]

    #     # Load Model
    #     model = get_my_model(model_name=MODEL, num_classes = 10, checkpoint_path="").to(device)
    #     model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])

    #     pred_lst = []
    #     file_name_lst = []

    #     with torch.no_grad():
    #         for img, file_name in tqdm(test_dataloader):
    #             img = img.to(device)
    #             pred = model(img)
    #             print("logits size", pred.size())
    #             pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
    #             print("pred_lst size", torch.tensor(pred_lst).size())
    #             file_name_lst.extend(file_name)
        
    #     total_pred.append(pred_lst)

    # pred_lst = torch.round(torch.mean(torch.FloatTensor(total_pred), 0)).tolist()

    # df = pd.DataFrame({'file_name':file_name_lst, 'answer':pred_lst})
    # df.to_csv(SAVE_PATH, index=None)



    # # Load Multi Model -> Ensemble with softmax scores
    # total_logits = []
    # for idx in range(len(model_list)):
    #     MODEL = model_list[idx]
    #     TRAINED_MODEL_PATH = model_path_list[idx]

    #     # Load Model
    #     model = get_my_model(model_name=MODEL, num_classes = 10, checkpoint_path="").to(device)
    #     model.load_state_dict(torch.load(TRAINED_MODEL_PATH)['model'])

    #     logits = []
    #     file_name_lst = []

    #     with torch.no_grad():
    #         for img, file_name in tqdm(test_dataloader):
    #             img = img.to(device)
    #             pred = model(img)
    #             logits.extend(pred.tolist())
    #             print("logits size: ", pred.size())
    #             file_name_lst.extend(file_name)
        
    #     total_logits.append(logits)
    #     print("total logits size: ", torch.tensor(total_logits).size())
    
    # total_logits_tensor = torch.FloatTensor(total_logits)
    # print("total_logits_tensor:", total_logits_tensor.size())
    # total_pred = torch.sum(total_logits_tensor, 0).argmax(dim=1)
    # print("total_pred size", total_pred.size())
    # pred_lst = total_pred.tolist()

    # df = pd.DataFrame({'file_name':file_name_lst, 'answer':pred_lst})
    # df.to_csv(SAVE_PATH, index=None)

