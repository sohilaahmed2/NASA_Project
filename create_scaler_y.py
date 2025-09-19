from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

y_dummy = np.array([
    [1.0, 5.0],
    [2.0, 10.0],
    [3.0, 15.0]
])

scaler_y = MinMaxScaler()
scaler_y.fit(y_dummy)

joblib.dump(scaler_y, "scaler_y.pkl")
print("✅ تم إنشاء الملف scaler_y.pkl بنجاح")
