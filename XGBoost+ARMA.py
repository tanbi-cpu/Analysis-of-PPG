import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings

# 忽略一些版本兼容性的警告，让输出更清爽
warnings.filterwarnings("ignore")

# --- 1. 数据加载与预处理 ---
print("正在加载和清洗数据...")
df = pd.read_csv('data_2012_2024.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 特征工程: 填充缺失值
feature_cols = [
    'Weather_Temperature_Celsius', 'Weather_Relative_Humidity',
    'Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation',
    'Wind_Direction', 'Weather_Daily_Rainfall',
    'Radiation_Global_Tilted', 'Radiation_Diffuse_Tilted'
]

# 删除目标变量缺失的行
df = df.dropna(subset=['Active_Power'])
# 确保数值类型
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 【修复 1】: 使用新版 Pandas 的写法 .ffill() 和 .bfill() 替代 method='...'
df[feature_cols] = df[feature_cols].interpolate(method='linear').ffill().bfill()

# 时间特征编码
df['hour'] = df['timestamp'].dt.hour
df['month'] = df['timestamp'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

input_features = feature_cols + ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
target = 'Active_Power'

# 划分数据集 (前80%训练, 后20%测试)
# 重置索引，保证索引是连续的整数，避免 statsmodels 报错
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size].reset_index(drop=True)
test_df = df.iloc[train_size:].reset_index(drop=True)

# ==========================================
# 第一阶段: Base Model (NWP -> Power)
# ==========================================
print("Step 1: Training Base Model (XGBoost/GBDT)...")
gb_model = HistGradientBoostingRegressor(max_iter=100, max_depth=8, learning_rate=0.1, random_state=42)
gb_model.fit(train_df[input_features], train_df[target])

# 预测并应用物理约束
train_df['pred_base'] = np.clip(gb_model.predict(train_df[input_features]), 0, None)
test_df['pred_base'] = np.clip(gb_model.predict(test_df[input_features]), 0, None)

# 计算残差
train_df['residual'] = train_df[target] - train_df['pred_base']
test_df['residual'] = test_df[target] - test_df['pred_base']

rmse_base = np.sqrt(mean_squared_error(test_df[target], test_df['pred_base']))
print(f"Stage 1 RMSE: {rmse_base:.4f}")

# ==========================================
# 第二阶段: Error Correction (ARMA on Residuals)
# ==========================================
print("\nStep 2: Training Error Correction Model (ARMA)...")

# 取训练集最后 50,000 个点
# 【修复 2】: 确保输入是纯 Series 且索引重置，避免 ValueWarning
train_resid_sample = train_df['residual'].iloc[-50000:].reset_index(drop=True)

# 建立 ARMA(4,1) 模型
arma_model = ARIMA(train_resid_sample, order=(4, 0, 1))
arma_result = arma_model.fit()

print("ARMA 模型拟合完成。")

print("\nApplying ARMA correction to Test Set...")
# 同样，为了避免索引对齐警告，这里我们提取 values 进行计算
test_residuals = test_df['residual'].reset_index(drop=True)
test_resid_model = arma_result.apply(test_residuals)
test_df['pred_residual'] = test_resid_model.fittedvalues.values # 取 values 赋值

# ==========================================
# 最终融合
# ==========================================
test_df['pred_final'] = test_df['pred_base'] + test_df['pred_residual']
test_df['pred_final'] = np.clip(test_df['pred_final'], 0, None)

# 评估
rmse_final = np.sqrt(mean_squared_error(test_df[target], test_df['pred_final']))
improvement = (rmse_base - rmse_final) / rmse_base * 100

print(f"\nFinal Results:")
print(f"Stage 1 (NWP Only) RMSE: {rmse_base:.4f}")
print(f"Stage 2 (Hybrid)   RMSE: {rmse_final:.4f}")
print(f"Improvement: {improvement:.2f}%")

# 可视化前300个点
plt.figure(figsize=(15, 6))
subset = test_df.iloc[400:1400]
plt.plot(subset.index, subset[target], label='Actual', color='black')
plt.plot(subset.index, subset['pred_base'], label='Stage 1 (NWP)', linestyle='--', color='blue')
plt.plot(subset.index, subset['pred_final'], label='Stage 2 (Hybrid)', color='red')
plt.legend()
plt.title('Comparison: Actual vs Stage 1 vs Stage 2')
plt.savefig('Comparison_Actual_Stage _1_Stage_2_2.png')
plt.show()