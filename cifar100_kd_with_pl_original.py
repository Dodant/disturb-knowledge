import argparse
import time
import warnings
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pl_bolts.callbacks import PrintTableMetricsCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import MetricCollection, MeanMetric, Accuracy

warnings.filterwarnings(action='ignore')
pl.seed_everything(42)


class ResNet_KD(pl.LightningModule):
    def __init__(self, student_model: int, T, alpha, batch):
        super(ResNet_KD, self).__init__()
        self.save_hyperparameters()
        model_link = 'chenyaofo/pytorch-cifar-models'
        self.teacher = torch.hub.load(model_link, 'cifar100_resnet56', pretrained=True)
        self.student = torch.hub.load(model_link, f'cifar100_resnet{student_model}', pretrained=False)

        metrics = MetricCollection({'loss': MeanMetric(), 'acc1': Accuracy(), 'acc5': Accuracy(top_k=5)})
        self.train_metrics, self.valid_metrics = metrics.clone(), metrics.clone()
        self.best_acc1, self.best_acc5, self.best_epoch = 0.0, 0.0, 0

        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.configure_optimizers(), T_max=400)
        # 수동으로 떨어트려라.

    def forward(self, x):
        return self.student(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    def distillation_loss(self, student_logit, teacher_logit, T):
        return F.kl_div(F.log_softmax(student_logit / T, dim=1), F.softmax(teacher_logit / T, dim=1))

    def FinalLoss(self, student_logit, teacher_logit, labels, T):
        return self.hparams.alpha * F.cross_entropy(student_logit, labels) + \
               (1. - self.hparams.alpha) * T * T * self.distillation_loss(student_logit, teacher_logit, T)

    # STEP
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        self.teacher.eval()
        with torch.no_grad():
            teacher_logit = self.teacher(inputs)
        student_logit = self.student(inputs)
        loss = self.FinalLoss(student_logit, teacher_logit, labels, self.hparams.T)
        self.train_metrics['loss'](loss)
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            self.sched.step()
        return {'loss': loss,
                'progress_bar':
                    {'top1_acc': self.train_metrics['acc1'](student_logit, labels),
                     'top5_acc': self.train_metrics['acc5'](student_logit, labels)}
                }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        student_logit, teacher_logit = self.student(inputs), self.teacher(inputs)
        loss = self.FinalLoss(student_logit, teacher_logit, labels, self.hparams.T)
        self.valid_metrics['loss'](loss)
        return {'loss': loss,
                'progress_bar':
                    {'top1_acc': self.valid_metrics['acc1'](student_logit, labels),
                     'top5_acc': self.valid_metrics['acc5'](student_logit, labels)}
                }

    # def test_step(self, batch, batch_idx):
    #     return self.validation_step(batch, batch_idx)

    # EPOCH_END
    def training_epoch_end(self, outputs):
        train_loss = self.train_metrics['loss'].compute()
        train_acc1, train_acc5 = self.train_metrics['acc1'].compute(), self.train_metrics['acc5'].compute()
        self.train_metrics.reset()
        self.log_dict({'Train Loss': train_loss,
                       'Train Top-1 Accuracy': train_acc1 * 100,
                       'Train Top-5 Accuracy': train_acc5 * 100,
                       'step': self.current_epoch + 1})

    def validation_epoch_end(self, outputs):
        valid_loss = self.valid_metrics['loss'].compute()
        valid_acc1, valid_acc5 = self.valid_metrics['acc1'].compute(), self.valid_metrics['acc5'].compute()
        self.valid_metrics.reset()
        self.log_dict({'Validation Loss': valid_loss,
                       'Validation Top-1 Accuracy': valid_acc1 * 100,
                       'Validation Top-5 Accuracy': valid_acc5 * 100,
                       'step': self.current_epoch + 1})
        # TODO more condition
        if self.best_acc1 < valid_acc1:
            self.best_acc1, self.best_acc5, self.best_epoch = valid_acc1, valid_acc5, self.current_epoch + 1

    # def test_epoch_end(self, outputs):
    #     test_acc1 = torch.stack([x['progress_bar']['top1_acc'] for x in outputs]).mean()
    #     test_acc5 = torch.stack([x['progress_bar']['top5_acc'] for x in outputs]).mean()
    #     self.log_dict({'Test Top-1 Accuracy': test_acc1 * 100,
    #                    'Test Top-5 Accuracy': test_acc5 * 100,
    #                    'step': 0.0})

    # DATASET, DATALOADER
    def prepare_data(self):
        datasets.CIFAR100(root='./data', train=True, download=True)
        datasets.CIFAR100(root='./data', train=False, download=True)

    def setup(self, stage=None):
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.RandomCrop((32, 32), padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.507, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761))])

        self.dataset_train = datasets.CIFAR100(root='./data', train=True, transform=train_transform)
        self.dataset_valid = datasets.CIFAR100(root='./data', train=False, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.hparams.batch, num_workers=4, pin_memory=True,
                          shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.hparams.batch, num_workers=4, pin_memory=True)


def kd_train_and_test(student_model, T, alpha, batch, epochs, repeat):
    acc1_list, acc5_list = [], []
    f = open(f"log_{alpha}_{T}.txt", 'a')
    config_str = '============================================================\nCONFIG\n' \
                 f'student_model = {student_model} | alpha = {alpha}'
    print(config_str)
    f.write(config_str + '\n')
    f.close()
    for i in range(repeat):
        f = open(f"log_{alpha}_{T}.txt", 'a')
        trial_str = f'------------------------------------------------------------\nTRIAL - {i + 1}'
        print(trial_str)
        f.write(trial_str + '\n')
        st = time.time()
        model = ResNet_KD(student_model, T, alpha, batch)
        logger = pl.loggers.TensorBoardLogger('tb_logs', name=f'baseline_{student_model}', version=f'alpha_{alpha}|{T}')
        # callback = PrintTableMetricsCallback()
        trainer = pl.Trainer(gpus=-1,
                             max_epochs=epochs,
                             weights_summary=None,
                             logger=logger,
                             # callbacks=[EarlyStopping(monitor="Validation Loss", mode="min")]
                             # callbacks=[callback]
                             )
        trainer.fit(model)
        acc1_list.append(best_acc1 := model.best_acc1.item() * 100)
        acc5_list.append(best_acc5 := model.best_acc5.item() * 100)
        result_str = f'Total training time: {(time.time() - st) / 60: .1f}m\n'\
                     f'Best Top1 Acc: {best_acc1:.2f}% - epoch : {model.best_epoch}\n'\
                     f'Best Top5 Acc: {best_acc5:.2f}%\n'
        print(result_str)
        f.write(result_str)
        f.close()

    f = open(f"log_{alpha}_{T}.txt", 'a')
    summary_str = f'Top1 Acc - best (avg±std): {max(acc1_list):.2f} ({mean(acc1_list):.2f}±{stdev(acc1_list):.2f})\n'\
                  f'Top5 Acc - best (avg±std): {max(acc5_list):.2f} ({mean(acc5_list):.2f}±{stdev(acc5_list):.2f})\n\n'
    print(summary_str)
    f.write(summary_str)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-student', type=int)
    parser.add_argument('-t', type=float)
    parser.add_argument('-alpha', type=float)

    parser.add_argument('-batch', type=int)
    parser.add_argument('-epochs', type=int)

    parser.add_argument('-repeat', type=int)

    args = parser.parse_args()
    kd_train_and_test(args.student, args.t, args.alpha, args.batch, args.epochs, args.repeat)
