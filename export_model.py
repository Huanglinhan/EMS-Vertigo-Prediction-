import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

print("开始读取本地数据并训练模型...")
# 1. 读取本地敏感数据
data_path = r"D:\OneDrive\Desktop\database_10_26_imputed.xlsx"
df = pd.read_excel(data_path)
X = df.iloc[:, :-1]
y = df.iloc[:, -1] 

features = ['VA-visibility', 'ED intensity', 'ATVA', 'QPVAA', 'PTA', 'Vertigo attack']
X_sub = X[features]

# 2. 训练模型
base_xgb = XGBClassifier(
    scale_pos_weight=(y==0).sum()/(y==1).sum(), 
    eval_metric="logloss", random_state=42
)
base_xgb.fit(X_sub, y)

calibrated_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv='prefit')
calibrated_xgb.fit(X_sub, y)

# 3. 固化并导出模型 (不包含任何原始数据)
output_dir = r"D:\OneDrive\Desktop\\"
joblib.dump(calibrated_xgb, output_dir + 'calibrated_model.pkl')
joblib.dump(base_xgb, output_dir + 'base_model.pkl')

print("✅ 模型打包成功！已在桌面生成 calibrated_model.pkl 和 base_model.pkl")
print("你的 Excel 数据可以安全地留在本地了！")