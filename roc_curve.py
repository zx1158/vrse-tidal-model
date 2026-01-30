# plot_roc.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 模拟数据（基于论文描述）
np.random.seed(42)
n = 1000
y_true = np.random.binomial(1, 0.3, n)  # 30% 阳性率

# 修正：为每个样本单独生成预测概率
n_pos = np.sum(y_true == 1)
n_neg = np.sum(y_true == 0)

# 为正负样本分别生成预测值
pred_vrse_pos = np.random.beta(8, 2, n_pos)  # VR-BETM对正样本的预测
pred_vrse_neg = np.random.beta(2, 8, n_neg)  # VR-BETM对负样本的预测
pred_base_pos = np.random.beta(6, 4, n_pos)  # Baseline对正样本的预测
pred_base_neg = np.random.beta(4, 6, n_neg)  # Baseline对负样本的预测

# 组合预测结果
y_pred_vrse = np.zeros(n)
y_pred_baseline = np.zeros(n)

y_pred_vrse[y_true == 1] = pred_vrse_pos
y_pred_vrse[y_true == 0] = pred_vrse_neg
y_pred_baseline[y_true == 1] = pred_base_pos
y_pred_baseline[y_true == 0] = pred_base_neg

# 计算 ROC
fpr_vrse, tpr_vrse, _ = roc_curve(y_true, y_pred_vrse)
fpr_base, tpr_base, _ = roc_curve(y_true, y_pred_baseline)

auc_vrse = auc(fpr_vrse, tpr_vrse)
auc_base = auc(fpr_base, tpr_base)

# 绘图
plt.figure(figsize=(6, 5))
plt.plot(fpr_vrse, tpr_vrse, label=f'VR-BETM (AUC = {auc_vrse:.3f})', linewidth=2)
plt.plot(fpr_base, tpr_base, label=f'Baseline (AUC = {auc_base:.3f})', linestyle='--')
plt.plot([0, 1], [0, 1], 'k:', alpha=0.5, label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()