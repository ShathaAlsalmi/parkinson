import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="Parkinson's Disease Prediction API",
    description="FastAPI service for early detection of Parkinson's disease using simplified voice acoustic biomarkers.",
    version="2.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "model.pkl"))
SCALER_PATH = os.getenv("SCALER_PATH", str(BASE_DIR / "scaler_params.json"))

# Exact ordered list of features expected by the 22-feature UCI Parkinson's dataset
FEATURE_NAMES: List[str] = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

# Baseline population means extracted directly from training dataset
DEFAULT_MEANS: Dict[str, float] = {
    "MDVP:Fo(Hz)": 154.22864,
    "MDVP:Fhi(Hz)": 197.10492,
    "MDVP:Flo(Hz)": 116.32463,
    "MDVP:Jitter(%)": 0.00622,
    "MDVP:Jitter(Abs)": 0.00004,
    "MDVP:RAP": 0.00331,
    "MDVP:PPQ": 0.00345,
    "Jitter:DDP": 0.00992,
    "MDVP:Shimmer": 0.02971,
    "MDVP:Shimmer(dB)": 0.28225,
    "Shimmer:APQ3": 0.01566,
    "Shimmer:APQ5": 0.01788,
    "MDVP:APQ": 0.02408,
    "Shimmer:DDA": 0.04699,
    "NHR": 0.02485,
    "HNR": 21.88597,
    "RPDE": 0.49854,
    "DFA": 0.71810,
    "spread1": -5.68440,
    "spread2": 0.22651,
    "D2": 2.38183,
    "PPE": 0.20655,
}

# Normative reference baselines for healthy control cohort vs parkinsonian dysphonia
HEALTHY_COHORT_MEANS: Dict[str, float] = {
    "MDVP:Fo(Hz)": 181.9378,
    "MDVP:Fhi(Hz)": 223.6368,
    "MDVP:Flo(Hz)": 145.2073,
    "MDVP:Jitter(%)": 0.003866,
    "MDVP:Jitter(Abs)": 0.0000234,
    "MDVP:RAP": 0.001925,
    "MDVP:PPQ": 0.002056,
    "Jitter:DDP": 0.005776,
    "MDVP:Shimmer": 0.017615,
    "MDVP:Shimmer(dB)": 0.162958,
    "Shimmer:APQ3": 0.009504,
    "Shimmer:APQ5": 0.010509,
    "MDVP:APQ": 0.013305,
    "Shimmer:DDA": 0.028511,
    "NHR": 0.011483,
    "HNR": 24.67875,
    "RPDE": 0.442552,
    "DFA": 0.695716,
    "spread1": -6.759264,
    "spread2": 0.160292,
    "D2": 2.154491,
    "PPE": 0.123017,
}

PD_COHORT_MEANS: Dict[str, float] = {
    "MDVP:Fo(Hz)": 145.1808,
    "MDVP:Fhi(Hz)": 188.4415,
    "MDVP:Flo(Hz)": 106.8936,
    "MDVP:Jitter(%)": 0.006989,
    "MDVP:Jitter(Abs)": 0.0000507,
    "MDVP:RAP": 0.003757,
    "MDVP:PPQ": 0.003900,
    "Jitter:DDP": 0.011273,
    "MDVP:Shimmer": 0.033658,
    "MDVP:Shimmer(dB)": 0.321204,
    "Shimmer:APQ3": 0.017676,
    "Shimmer:APQ5": 0.020285,
    "MDVP:APQ": 0.027600,
    "Shimmer:DDA": 0.053027,
    "NHR": 0.029211,
    "HNR": 20.97405,
    "RPDE": 0.516816,
    "DFA": 0.725408,
    "spread1": -5.333420,
    "spread2": 0.248133,
    "D2": 2.456058,
    "PPE": 0.233828,
}

DEFAULT_STDS: Dict[str, float] = {
    "MDVP:Fo(Hz)": 41.39006,
    "MDVP:Fhi(Hz)": 91.49155,
    "MDVP:Flo(Hz)": 43.52141,
    "MDVP:Jitter(%)": 0.004848,
    "MDVP:Jitter(Abs)": 0.00003482,
    "MDVP:RAP": 0.002968,
    "MDVP:PPQ": 0.002759,
    "Jitter:DDP": 0.008903,
    "MDVP:Shimmer": 0.018857,
    "MDVP:Shimmer(dB)": 0.194877,
    "Shimmer:APQ3": 0.010153,
    "Shimmer:APQ5": 0.012024,
    "MDVP:APQ": 0.016947,
    "Shimmer:DDA": 0.030459,
    "NHR": 0.040418,
    "HNR": 4.425764,
    "RPDE": 0.103942,
    "DFA": 0.055336,
    "spread1": 1.090208,
    "spread2": 0.083406,
    "D2": 0.382799,
    "PPE": 0.090119,
}

# Load scaler if available
feature_means = np.array([DEFAULT_MEANS[col] for col in FEATURE_NAMES], dtype=np.float64)
feature_stds = np.array([DEFAULT_STDS[col] for col in FEATURE_NAMES], dtype=np.float64)

if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "r") as f:
            data = json.load(f)
            if "means" in data and "stds" in data:
                feature_means = np.array([data["means"][col] for col in FEATURE_NAMES], dtype=np.float64)
                feature_stds = np.array([data["stds"][col] for col in FEATURE_NAMES], dtype=np.float64)
                print("[*] Loaded scaler parameters from scaler_params.json")
    except Exception as e:
        print(f"[!] Warning reading scaler_params.json: {e}")

# Load model safely at startup
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"[*] Successfully loaded Parkinson's model from: {MODEL_PATH}")
    else:
        print(f"[!] Warning: Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"[!] Error loading model: {e}")


class PatientFeatures(BaseModel):
    """
    الخصائص الخمس الأساسية فقط التي يطلبها المستخدم في الواجهة المبسطة.
    باقي الـ 17 خاصية سيتم إكمالها تلقائياً بالمتوسطات في الخلفية.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    ppe: float = Field(..., alias="PPE", description="Pitch period entropy")
    jitter_percent: float = Field(..., alias="MDVP:Jitter(%)", description="MDVP jitter percentage")
    shimmer_apq5: float = Field(..., alias="Shimmer:APQ5", description="Five-point amplitude perturbation quotient")
    shimmer_dda: float = Field(..., alias="Shimmer:DDA", description="Average absolute differences between amplitudes")
    shimmer_apq3: float = Field(..., alias="Shimmer:APQ3", description="Three-point amplitude perturbation quotient")

    def to_feature_vector(self) -> List[float]:
        """دمج القيم الخمس مع المتوسطات الافتراضية وترتيبها بالترتيب الصحيح للنموذج."""
        # Calculate physiological indicator r based on the 5 clinical voice features (0.0=healthy, 1.0=PD)
        r_ppe = (float(self.ppe) - 0.10) / (0.24 - 0.10)
        r_jit = (float(self.jitter_percent) - 0.003) / (0.007 - 0.003)
        r_shimmer = (float(self.shimmer_apq5) - 0.010) / (0.021 - 0.010)
        r = float(np.clip(0.4 * r_ppe + 0.3 * r_jit + 0.3 * r_shimmer, 0.0, 1.0))

        complete_features = {}
        for col in FEATURE_NAMES:
            # Baseline interpolation between reference normative averages and dysphonia averages
            complete_features[col] = (1.0 - r) * HEALTHY_COHORT_MEANS.get(col, DEFAULT_MEANS[col]) + r * PD_COHORT_MEANS.get(col, DEFAULT_MEANS[col])

        # تحديث القيم بالمدخلات الفعلية من المستخدم
        complete_features["PPE"] = float(self.ppe)
        complete_features["MDVP:Jitter(%)"] = float(self.jitter_percent)
        complete_features["Shimmer:APQ5"] = float(self.shimmer_apq5)
        complete_features["Shimmer:DDA"] = float(self.shimmer_dda)
        complete_features["Shimmer:APQ3"] = float(self.shimmer_apq3)
        
        # إرجاعها مصفوفة مرتبة حسب FEATURE_NAMES بالكامل
        return [float(complete_features[col]) for col in FEATURE_NAMES]


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Negative / Healthy, 1 = Positive / Parkinson's Detected")
    label: str = Field(..., description="Human-readable diagnosis result")
    risk_level: str = Field(..., description="Categorized risk assessment (Low, Moderate, High)")
    probability: float = Field(..., description="Confidence probability for the predicted class (0.0 to 1.0)")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for Healthy (0) and Parkinson's (1)")
    features_received: Dict[str, float] = Field(..., description="Parsed input features mapped back to standard clinical keys")


@app.get("/")
def root():
    return {
        "service": "Parkinson's Disease Prediction API (Simplified UI)",
        "status": "online",
        "model_loaded": model is not None,
        "n_features_required_by_model": len(FEATURE_NAMES),
        "n_features_exposed_in_ui": 5,
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure model.pkl exists in the backend directory."
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    """
    يستقبل الخصائص الخمس فقط، يكمل الباقي بالمتوسطات، ويقوم بالتنبؤ عبر النموذج.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive model is not available. Please verify backend/model.pkl exists."
        )

    # تحويل البيانات إلى متجه الـ 22 خاصية كاملة
    raw_vals = np.array(features.to_feature_vector(), dtype=np.float64)

    # معيار التوحيد القياسي للنموذج
    if raw_vals[0] > 10.0:
        scaled_vals = (raw_vals - feature_means) / feature_stds
        X = scaled_vals.reshape(1, -1)
    else:
        X = raw_vals.reshape(1, -1)

    try:
        pred_raw = model.predict(X)[0]
        prediction_val = int(pred_raw)

        prob_healthy = 0.5
        prob_parkinsons = 0.5

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            if len(probs) >= 2:
                prob_healthy = float(probs[0])
                prob_parkinsons = float(probs[1])
            else:
                prob_parkinsons = float(probs[0])
                prob_healthy = 1.0 - prob_parkinsons
        elif hasattr(model, "decision_function"):
            df_val = float(model.decision_function(X)[0])
            prob_parkinsons = float(1.0 / (1.0 + np.exp(-df_val)))
            prob_healthy = float(1.0 - prob_parkinsons)

        if prob_parkinsons >= 0.5:
            prediction_val = 1
        else:
            prediction_val = 0

        if prediction_val == 1:
            label = "Parkinson's Disease Detected"
            predicted_prob = prob_parkinsons
            risk_level = "High Risk" if predicted_prob >= 0.80 else "Moderate Risk"
        else:
            label = "Healthy / Low Risk"
            predicted_prob = prob_healthy
            risk_level = "Low Risk (Normal)" if predicted_prob >= 0.75 else "Mild Indicator"

        received_dict = {name: val for name, val in zip(FEATURE_NAMES, raw_vals)}

        return PredictionResponse(
            prediction=prediction_val,
            label=label,
            risk_level=risk_level,
            probability=round(predicted_prob, 4),
            class_probabilities={
                "Healthy": round(prob_healthy, 4),
                "Parkinsons": round(prob_parkinsons, 4),
            },
            features_received=received_dict,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during model prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
