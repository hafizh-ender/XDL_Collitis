import torch
import numpy as np
from src.metrics import (
    mcc_score,
    CustomPrecision,
    CustomRecall,
    CustomAUC,
    CustomSpecificity
)

def create_dummy_data(num_classes=3, batch_size=10, seed=42):
    """Create dummy data for testing metrics"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create true labels and predictions with some errors
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # True labels
    y_pred = torch.tensor([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])  # Predicted labels (with some errors)
    
    # Convert to one-hot encoding
    y_true_onehot = torch.zeros(batch_size, num_classes)
    y_true_onehot[torch.arange(batch_size), y_true] = 1
    
    y_pred_onehot = torch.zeros(batch_size, num_classes)
    y_pred_onehot[torch.arange(batch_size), y_pred] = 1
    
    # Create probability predictions for AUC
    y_pred_probs = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    
    return y_true, y_pred, y_true_onehot, y_pred_onehot, y_pred_probs

def test_mcc():
    """Test Matthews Correlation Coefficient"""
    print("\n=== Testing MCC Score ===")
    y_true, y_pred, y_true_onehot, y_pred_onehot, _ = create_dummy_data()
    
    # Calculate MCC
    mcc = mcc_score(y_true_onehot, y_pred_onehot)
    print(f"MCC Score: {mcc:.4f}")
    print("Expected: Value between -1 and 1, where:")
    print("  1: Perfect prediction")
    print("  0: Random prediction")
    print(" -1: Perfect inverse prediction")

def test_precision_recall_specificity():
    """Test Precision, Recall, and Specificity with different averaging methods"""
    print("\n=== Testing Precision, Recall, and Specificity ===")
    y_true, y_pred, y_true_onehot, y_pred_onehot, _ = create_dummy_data()
    
    metrics = {
        'Precision': CustomPrecision(num_classes=3),
        'Recall': CustomRecall(num_classes=3),
        'Specificity': CustomSpecificity(num_classes=3)
    }
    
    for metric_name, metric in metrics.items():
        print(f"\n{metric_name}:")
        for avg in ['macro', 'micro', 'weighted', 'none']:
            metric.average = avg
            metric.reset()
            metric.update(y_true_onehot, y_pred_onehot)
            result = metric.compute()
            
            if isinstance(result, torch.Tensor) and result.numel() > 1:
                print(f"{avg.capitalize()}: {result.tolist()}")
            else:
                print(f"{avg.capitalize()}: {result:.4f}")
        
        print(f"\nExplanation for {metric_name}:")
        if metric_name == 'Precision':
            print("Precision measures the accuracy of positive predictions")
        elif metric_name == 'Recall':
            print("Recall measures the ability to find all positive cases")
        elif metric_name == 'Specificity':
            print("Specificity measures the ability to find all negative cases")

def test_auc():
    """Test Area Under Curve (AUC)"""
    print("\n=== Testing AUC ===")
    y_true, y_pred, y_true_onehot, y_pred_onehot, y_pred_probs = create_dummy_data()
    
    # Test macro AUC
    auc = CustomAUC(num_classes=3)
    auc.update(y_true_onehot, y_pred_probs)
    macro_auc = auc.compute()
    print(f"Macro AUC: {macro_auc:.4f}")
    
    # Test per-class AUC
    auc.average = 'none'
    auc.reset()
    auc.update(y_true_onehot, y_pred_probs)
    per_class_auc = auc.compute()
    print(f"Per-class AUC: {per_class_auc.tolist()}")
    
    print("\nExplanation:")
    print("AUC measures the model's ability to distinguish between classes")
    print("Values range from 0 to 1, where:")
    print("  1: Perfect discrimination")
    print("  0.5: Random discrimination")
    print("  0: Perfect inverse discrimination")

def test_edge_cases():
    """Test metrics with edge cases"""
    print("\n=== Testing Edge Cases ===")
    
    # Test with perfect predictions
    print("\nPerfect Predictions:")
    y_true = torch.tensor([0, 1, 2, 0, 1])
    y_pred = torch.tensor([0, 1, 2, 0, 1])
    y_true_onehot = torch.zeros(5, 3)
    y_true_onehot[torch.arange(5), y_true] = 1
    y_pred_onehot = torch.zeros(5, 3)
    y_pred_onehot[torch.arange(5), y_pred] = 1
    
    mcc = mcc_score(y_true_onehot, y_pred_onehot)
    print(f"MCC Score: {mcc:.4f} (Should be 1.0)")
    
    # Test with random predictions
    print("\nRandom Predictions:")
    y_pred = torch.randint(0, 3, (5,))
    y_pred_onehot = torch.zeros(5, 3)
    y_pred_onehot[torch.arange(5), y_pred] = 1
    
    mcc = mcc_score(y_true_onehot, y_pred_onehot)
    print(f"MCC Score: {mcc:.4f} (Should be close to 0)")

def main():
    """Run all tests"""
    print("Starting metrics tests...")
    print("=" * 50)
    
    test_mcc()
    test_precision_recall_specificity()
    test_auc()
    test_edge_cases()
    
    print("\nAll tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main() 