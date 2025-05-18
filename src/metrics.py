import torch

def mcc_score(y_true, y_pred):
    """
    Matthew Correlation Coefficient (MCC) for multiclass
    """
    # Get predicted class indices
    y_true_label = torch.argmax(y_true, dim=1)
    y_pred_label = torch.argmax(y_pred, dim=1)

    # Get number of classes
    num_classes = y_pred.size(1)

    # Compute confusion matrix
    cm = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=y_pred.device)
    for t, p in zip(y_true_label, y_pred_label):
        cm[t.item(), p.item()] += 1

    # Compute necessary sums
    t = torch.sum(cm, dim=1)  # True counts per class
    p = torch.sum(cm, dim=0)  # Predicted counts per class
    c = torch.diag(cm)  # Correctly predicted per class
    s = torch.sum(cm)  # Total samples

    # Compute MCC numerator and denominator
    numerator = torch.sum(c) * s - torch.dot(t, p)
    denominator = torch.sqrt((s ** 2 - torch.dot(p, p)) * (s ** 2 - torch.dot(t, t)))

    return numerator / (denominator + torch.finfo(torch.float32).eps)  # Avoid division by zero

# class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
#     def __init__(self, num_classes, name='matthews_correlation_coefficient', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.num_classes = num_classes
#         self.confusion_matrix = self.add_weight(
#             name='conf_matrix',
#             shape=(num_classes, num_classes),
#             initializer='zeros',
#             dtype=tf.float32
#         )

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true_labels = tf.argmax(y_true, axis=1, output_type=tf.int32)
#         y_pred_labels = tf.argmax(y_pred, axis=1, output_type=tf.int32)

#         cm = tf.math.confusion_matrix(
#             y_true_labels,
#             y_pred_labels,
#             num_classes=self.num_classes,
#             dtype=tf.float32
#         )

#         self.confusion_matrix.assign_add(cm)

#     def result(self):
#         cm = self.confusion_matrix

#         # Number of times a class truly occured
#         t = tf.reduce_sum(cm, axis=1) # 1 x num_classes

#         # Number of times a class is predicted
#         p = tf.reduce_sum(cm, axis=0) # 1 x num_classes array

#         # Total number of correctly predicted
#         c = tf.reduce_sum(tf.linalg.diag_part(cm)) # 1 x num_classes array

#         # Total number of samples
#         s = tf.reduce_sum(cm) # scalar

#         numerator = c * s - tf.tensordot(t, p, axes=1)
#         denominator = tf.sqrt((s**2 - tf.tensordot(p, p, axes=1)) * (s**2 - tf.tensordot(t, t, axes=1)))

#         return numerator / (denominator + tf.keras.backend.epsilon())

#     def reset_states(self):
#         self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

class CustomPrecision(torch.nn.Module):
    def __init__(self, num_classes, average="macro"):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.register_buffer('confusion_matrix', torch.zeros(num_classes, num_classes))

    def update(self, y_true, y_pred):
        y_true_labels = torch.argmax(y_true, dim=1)
        y_pred_labels = torch.argmax(y_pred, dim=1)

        cm = torch.zeros(self.num_classes, self.num_classes, device=y_true.device)
        for t, p in zip(y_true_labels, y_pred_labels):
            cm[t, p] += 1

        self.confusion_matrix += cm

    def compute(self):
        cm = self.confusion_matrix
        tp = torch.diag(cm)
        predicted_positives = torch.sum(cm, dim=0)
        actual_positives = torch.sum(cm, dim=1)
        precision_per_class = tp / (predicted_positives + torch.finfo(torch.float32).eps)

        if self.average == "macro":
            return torch.mean(precision_per_class)
        elif self.average == "micro":
            total_tp = torch.sum(tp)
            total_predicted_positives = torch.sum(predicted_positives)
            return total_tp / (total_predicted_positives + torch.finfo(torch.float32).eps)
        elif self.average == "weighted":
            weights = actual_positives / (torch.sum(actual_positives) + torch.finfo(torch.float32).eps)
            return torch.sum(precision_per_class * weights)
        elif self.average == "none":
            return precision_per_class
        else:
            raise ValueError(f"Unknown average type: {self.average}")

    def reset(self):
        self.confusion_matrix.zero_()

class CustomRecall(torch.nn.Module):
    def __init__(self, num_classes, average="macro"):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.register_buffer('confusion_matrix', torch.zeros(num_classes, num_classes))

    def update(self, y_true, y_pred):
        y_true_labels = torch.argmax(y_true, dim=1)
        y_pred_labels = torch.argmax(y_pred, dim=1)

        cm = torch.zeros(self.num_classes, self.num_classes, device=y_true.device)
        for t, p in zip(y_true_labels, y_pred_labels):
            cm[t, p] += 1

        self.confusion_matrix += cm

    def compute(self):
        cm = self.confusion_matrix
        tp = torch.diag(cm)
        actual_positives = torch.sum(cm, dim=1)
        recall_per_class = tp / (actual_positives + torch.finfo(torch.float32).eps)

        if self.average == "macro":
            return torch.mean(recall_per_class)
        elif self.average == "micro":
            total_tp = torch.sum(tp)
            total_actual_positives = torch.sum(actual_positives)
            return total_tp / (total_actual_positives + torch.finfo(torch.float32).eps)
        elif self.average == "weighted":
            weights = actual_positives / (torch.sum(actual_positives) + torch.finfo(torch.float32).eps)
            return torch.sum(recall_per_class * weights)
        elif self.average == "none":
            return recall_per_class
        else:
            raise ValueError(f"Unknown average type: {self.average}")

    def reset(self):
        self.confusion_matrix.zero_()

class CustomAUC(torch.nn.Module):
    def __init__(self, num_classes, average="macro"):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.register_buffer('all_targets', torch.tensor([]))
        self.register_buffer('all_preds', torch.tensor([]))

    def update(self, y_true, y_pred):
        self.all_targets = torch.cat([self.all_targets, y_true.detach().cpu()])
        self.all_preds = torch.cat([self.all_preds, y_pred.detach().cpu()])

    def compute(self):
        aucs = []
        for i in range(self.num_classes):
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(self.all_targets[:, i].numpy(), self.all_preds[:, i].numpy())
                aucs.append(auc)
            except:
                aucs.append(0.0)
        aucs = torch.tensor(aucs)

        if self.average == "macro":
            return torch.mean(aucs)
        elif self.average == "none":
            return aucs
        else:
            raise ValueError(f"Averaging method {self.average} is not supported for AUC (use 'macro' or 'none').")

    def reset(self):
        self.all_targets = torch.tensor([])
        self.all_preds = torch.tensor([])

class CustomSpecificity(torch.nn.Module):
    def __init__(self, num_classes, average="macro"):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.register_buffer('confusion_matrix', torch.zeros(num_classes, num_classes))

    def update(self, y_true, y_pred):
        y_true_labels = torch.argmax(y_true, dim=1)
        y_pred_labels = torch.argmax(y_pred, dim=1)

        cm = torch.zeros(self.num_classes, self.num_classes, device=y_true.device)
        for t, p in zip(y_true_labels, y_pred_labels):
            cm[t, p] += 1

        self.confusion_matrix += cm

    def compute(self):
        cm = self.confusion_matrix
        tp = torch.diag(cm)
        fp = torch.sum(cm, dim=0) - tp
        fn = torch.sum(cm, dim=1) - tp
        tn = torch.sum(cm) - (tp + fp + fn)

        specificity_per_class = tn / (tn + fp + torch.finfo(torch.float32).eps)

        if self.average == "macro":
            return torch.mean(specificity_per_class)
        elif self.average == "micro":
            total_tn = torch.sum(tn)
            total_fp = torch.sum(fp)
            return total_tn / (total_tn + total_fp + torch.finfo(torch.float32).eps)
        elif self.average == "weighted":
            actual_negatives = torch.sum(cm, dim=1)
            weights = actual_negatives / (torch.sum(actual_negatives) + torch.finfo(torch.float32).eps)
            return torch.sum(specificity_per_class * weights)
        elif self.average == "none":
            return specificity_per_class
        else:
            raise ValueError(f"Unknown average type: {self.average}")

    def reset(self):
        self.confusion_matrix.zero_()
