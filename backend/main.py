from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# إنشاء تطبيق FastAPI
app = FastAPI()

# إضافة Middleware لتمكين CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد نطاق معين بدلاً من السماح للجميع
    allow_credentials=True,
    allow_methods=["*"],  # السماح بكل الطرق (مثل POST, GET, OPTIONS)
    allow_headers=["*"],  # السماح بكل الهيدر
)

# تخزين مسارات النماذج
best_models = {
    "1min": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/1min_best_model_polynomial_regression.pkl",
    "5min": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/5min_best_model_random_forest.pkl",
    "15min": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/15min_best_model_random_forest.pkl",
    "30min": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/30min_best_model_random_forest.pkl",
    "60min": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/60min_best_model_random_forest.pkl",
    "daily": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/daily_best_model_random_forest.pkl",
    "weekly": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/weekly_best_model_random_forest.pkl",
    "monthly": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/monthly_best_model_random_forest.pkl",
    "all_frames": "C:/Users/user/Desktop/ML_project/Tesla_sock_price_predection/backend/models/best_model_all_frames.pkl",
}

# تعريف نموذج البيانات باستخدام Pydantic
class PredictionRequest(BaseModel):
    frame: str
    Open: float
    High: float
    Low: float
    Volume: float

@app.post("/predict/")
def predict(request: PredictionRequest):
    # التحقق من صحة الإطار الزمني
    if request.frame not in best_models:
        return {"error": "Invalid frame. Supported frames are: " + ", ".join(best_models.keys())}

    # تحميل النموذج المناسب للإطار الزمني المحدد
    model_path = best_models[request.frame]
    try:
        selected_model = joblib.load(model_path)
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    # إعداد البيانات للتنبؤ
    input_data = {
        "Open": request.Open,
        "High": request.High,
        "Low": request.Low,
        "Volume": request.Volume
    }

    try:
        df = pd.DataFrame([input_data])
    except Exception as e:
        return {"error": f"Failed to convert data to DataFrame: {str(e)}"}

    # تنفيذ التنبؤ
    try:
        prediction = selected_model.predict(df)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    return {"prediction": prediction.tolist()}

# نقطة فحص للتأكد من عمل التطبيق
@app.get("/health")
def health_check():
    return {"status": "OK"}
