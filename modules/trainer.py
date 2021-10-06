"""Trainer 클래스 정의

TODO:

NOTES:

REFERENCE:

UPDATED:
"""


import torch
from tqdm import tqdm


class Trainer():
    """ Trainer
        epoch에 대한 학습 및 검증 절차 정의
    """

    def __init__(self, criterion, model, device, metric_fn, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.criterion = criterion
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.metric_fn = metric_fn
        self.train_mean_loss = 0
        self.val_mean_loss = 0
        self.train_score = 0
        self.validation_score = 0

    def train_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.train()
        train_total_loss = 0
        target_lst = []
        pred_lst = []
        prob_lst = []
        batch_index = 0

        for img, label in tqdm(dataloader):
            batch_index += 1
            img = img.to(self.device)
            label = label.to(self.device).long()
            pred = self.model(img)
            loss = self.criterion(pred, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_total_loss += loss.item()
            # self.scheduler.step()
            prob_lst.extend(pred[:, 1].cpu().tolist())
            target_lst.extend(label.cpu().tolist())
            pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
        self.train_mean_loss = train_total_loss / batch_index
        # self.scheduler.step(self.train_mean_loss)  # for plateu scheduler
        self.train_score = self.metric_fn(y_pred=pred_lst, y_answer=target_lst, y_prob=prob_lst)
        msg = f'Epoch {epoch_index}, Train loss: {self.train_mean_loss}, Acc: {self.train_score}'
        print(msg)
        #self.logger.info(msg) if self.logger else print(msg)

    def validate_epoch(self, dataloader, epoch_index, mode=None):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        val_total_loss = 0
        target_lst = []
        pred_lst = []
        prob_lst = []
        with torch.no_grad():
            for batch_index, (img, label) in enumerate(dataloader):
                img = img.to(self.device)
                label = label.to(self.device).long()
                pred = self.model(img)
                loss = self.criterion(pred, label)
                val_total_loss += loss.item()
                prob_lst.extend(pred[:, 1].cpu().tolist())
                target_lst.extend(label.cpu().tolist())
                pred_lst.extend(pred.argmax(dim=1).cpu().tolist())
            self.val_mean_loss = val_total_loss / batch_index
            self.validation_score = self.metric_fn(y_pred=pred_lst, y_answer=target_lst, y_prob=prob_lst)
            msg = f'Epoch {epoch_index}, {mode} loss: {self.val_mean_loss}, Acc: {self.validation_score}'
            print(msg)
        #self.logger.info(msg) if self.logger else print(msg)


