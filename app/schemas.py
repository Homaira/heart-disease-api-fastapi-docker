from pydantic import BaseModel, Field


class HeartFeatures(BaseModel):
    """Input features for the heart disease classifier (UCI/Kaggle heart.csv)."""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    heart_disease: bool
    probability: float = Field(..., description="Model probability of heart disease")


class InfoResponse(BaseModel):
    model_type: str
    n_features: int
    features: list[str]
