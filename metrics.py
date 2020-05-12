import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
import core.base as base
import constants
from collections import Counter


class Metric:

    def __str__(self):
        raise NotImplementedError("A metric must implement the method '__str__'")

    def fit(self, embeddings, y):
        raise NotImplementedError("A metric must implement the method 'fit'")

    def calculate_batch(self, embeddings, logits, y):
        raise NotImplementedError("A metric must implement the method 'calculate_batch'")

    def get(self):
        raise NotImplementedError("A metric must implement the method 'get'")


class KNNF1ScoreMetric(Metric):

    def __init__(self, distance, neighbors: int = 1):
        self.neighbors = neighbors
        self.knn = KNeighborsClassifier(n_neighbors=neighbors, metric=distance.to_sklearn_metric())
        self.preds, self.y = [], []
        self.train_class_counter = None
        self.train_y = None

    def __str__(self):
        return 'KNN Macro F1-Score'

    def fit(self, embeddings, y):
        self.knn.fit(embeddings, y)
        self.train_y = y
        self.train_class_counter = Counter(y)

    def calculate_batch(self, embeddings, logits, y):
        if self.neighbors == 1:
            predicted = self.knn.predict(embeddings)
        else:
            _, idx = self.knn.kneighbors(embeddings)
            predicted = []
            for neigh_labels in self.train_y[idx]:
                counter = Counter(neigh_labels)
                max_vote, max_label = 0, -1
                for label in counter:
                    vote = counter[label] / self.train_class_counter[label]
                    if vote > max_vote:
                        max_vote = vote
                        max_label = label
                predicted.append(max_label)
            predicted = np.array(predicted)
        self.preds.extend(predicted)
        self.y.extend(y)

    def get(self):
        metric = f1_score(self.y, self.preds, average='macro')
        print(f"Confusion Matrix:\n{confusion_matrix(self.y, self.preds)}")
        self.preds, self.y = [], []
        return metric


class Evaluator(base.TrainingListener):

    def __init__(self, loader, metric, partition_name, callbacks=None):
        self.loader = loader
        self.metric = metric
        self.partition_name = partition_name
        self.callbacks = callbacks if callbacks is not None else []
        self.feat_train, self.y_train = None, None
        self.results = []
        self.best_metric, self.best_epoch, self.last_metric = 0, -1, 0

    def eval(self, model):
        model.eval()

        feat_test, logits_test, y_test = [], [], []
        for cb in self.callbacks:
            cb.on_before_test()

        with torch.no_grad():
            for i in range(self.loader.nbatches()):
                x, y = next(self.loader)

                if isinstance(x, torch.Tensor):
                    x = x.to(constants.DEVICE)
                if isinstance(y, torch.Tensor):
                    y = y.to(constants.DEVICE)

                # Feed Forward
                feat, logits = model(x, y)
                feat = feat.detach().cpu().numpy()
                if logits is not None:
                    logits = logits.detach().cpu().numpy()
                y = y.detach().cpu().numpy()

                # Track accuracy
                feat_test.append(feat)
                if logits is not None:
                    logits_test.append(logits)
                y_test.append(y)
                self.metric.calculate_batch(feat, logits, y)

                for cb in self.callbacks:
                    cb.on_batch_tested(i, feat)

        feat_test, y_test = np.concatenate(feat_test), np.concatenate(y_test)
        return feat_test, y_test

    def on_before_epoch(self, epoch):
        self.feat_train, self.y_train = [], []

    def on_after_gradients(self, epoch, ibatch, feat, logits, y, loss):
        self.feat_train.append(feat.detach().cpu().numpy())
        self.y_train.append(y.detach().cpu().numpy())

    def on_after_epoch(self, epoch, model, loss_fn, optim):
        feat_train = np.concatenate(self.feat_train)
        y_train = np.concatenate(self.y_train)
        self.metric.fit(feat_train, y_train)

        feat_test, y_test = self.eval(model)

        self.last_metric = self.metric.get()
        self.results.append(self.last_metric)

        for cb in self.callbacks:
            cb.on_after_test(epoch, feat_test, y_test, self.last_metric)

        print(f"[{self.partition_name.capitalize()} {self.metric}: {self.last_metric:.6f}]")

        if self.best_epoch != -1:
            print(f"Best until now: {self.best_metric:.6f}, at epoch {self.best_epoch}")

        if self.last_metric > self.best_metric:
            self.best_metric = self.last_metric
            self.best_epoch = epoch

            print(f'New Best {self.partition_name.capitalize()} {self.metric}!')

            for cb in self.callbacks:
                cb.on_best_accuracy(epoch, model, loss_fn, optim, self.last_metric, feat_test, y_test)
