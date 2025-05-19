import torch
from torcheval.metrics.metric import Metric
from typing import Iterable, TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from typing_extensions import Self # Or from typing import Self for Python 3.11+

_T = TypeVar("_T", bound="Metric") # TypeVar for merge_state

class MulticlassMCC(Metric):
    def __init__(self, num_classes, device: torch.device | str | None = None):
        resolved_device = device if isinstance(device, torch.device) else torch.device(device or 'cpu')
        super().__init__(device=resolved_device)
        self.num_classes = num_classes
        self._add_state("confusion_matrix", torch.zeros(num_classes, num_classes, device=resolved_device))

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> "Self":
        y_true_labels = y_true
        y_pred_labels = y_pred
        
        if self.confusion_matrix.device != y_true.device:
            self.confusion_matrix = self.confusion_matrix.to(y_true.device)
        
        for t, p in zip(y_true_labels, y_pred_labels):
            self.confusion_matrix[t.long(), p.long()] += 1

        return self

    def compute(self) -> torch.Tensor:
        cm = self.confusion_matrix
        t = torch.sum(cm, dim=1)
        p = torch.sum(cm, dim=0)
        c = torch.sum(torch.diag(cm))
        s = torch.sum(cm)
        numerator = c * s - torch.dot(t, p)
        denominator = torch.sqrt((s**2 - torch.dot(p, p)) * (s**2 - torch.dot(t, t)))
        return numerator / (denominator + torch.finfo(torch.float32).eps)

    def reset(self) -> "Self":
        self.confusion_matrix.zero_()
        return self

    def merge_state(self: _T, metrics: Iterable[_T]) -> _T:
        for metric_obj in metrics:
            if not isinstance(metric_obj, MulticlassMCC):
                raise TypeError(f"Cannot merge_state for MulticlassMCC with metric of type {type(metric_obj).__name__}")
            typed_metric = cast(MulticlassMCC, metric_obj)
            
            other_cm_state = typed_metric.confusion_matrix
            if self.confusion_matrix.device != other_cm_state.device:
                other_cm_state = other_cm_state.to(self.confusion_matrix.device)
            self.confusion_matrix += other_cm_state
        return self

class MulticlassSpecificity(Metric):
    def __init__(self, num_classes, device: torch.device | str | None = None, average="macro"):
        resolved_device = device if isinstance(device, torch.device) else torch.device(device or 'cpu')
        super().__init__(device=resolved_device)
        self.num_classes = num_classes
        self.average = average
        self._add_state("confusion_matrix", torch.zeros(num_classes, num_classes, device=resolved_device))

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> "Self":
        y_true_labels = y_true
        y_pred_labels = y_pred

        if self.confusion_matrix.device != y_true.device:
            self.confusion_matrix = self.confusion_matrix.to(y_true.device)
            
        for t, p in zip(y_true_labels, y_pred_labels):
            self.confusion_matrix[t.long(), p.long()] += 1
        
        return self

    def compute(self) -> torch.Tensor:
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
            support = torch.sum(cm, dim=0) - torch.diag(cm) + (torch.sum(cm) - torch.sum(cm, dim=1))
            weights = support / (torch.sum(support) + torch.finfo(torch.float32).eps)
            return torch.sum(specificity_per_class * weights)
        elif self.average == "none":
            return specificity_per_class
        else:
            raise ValueError(f"Unknown average type: {self.average}")

    def reset(self) -> "Self":
        self.confusion_matrix.zero_()
        return self

    def merge_state(self: _T, metrics: Iterable[_T]) -> _T:
        for metric_obj in metrics:
            if not isinstance(metric_obj, MulticlassSpecificity):
                raise TypeError(f"Cannot merge_state for MulticlassSpecificity with metric of type {type(metric_obj).__name__}")
            typed_metric = cast(MulticlassSpecificity, metric_obj)
            
            other_cm_state = typed_metric.confusion_matrix
            if self.confusion_matrix.device != other_cm_state.device:
                other_cm_state = other_cm_state.to(self.confusion_matrix.device)
            self.confusion_matrix += other_cm_state
        return self