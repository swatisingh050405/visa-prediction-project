from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from us_visa.exception import USvisaException
from us_visa.pipeline.prediction_pipeline import (
    PredictionPipeline,
    USvisaData
)
from us_visa.entity.prediction_schema import VisaRequest
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse



app = FastAPI(
    title="US Visa Prediction API",
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="templates/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# Exception Handlers

@app.exception_handler(USvisaException)
async def usvisa_exception_handler(request: Request, exc: USvisaException):

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc)
        }
    )




@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": str(exc)
        }
    )



# Health Check
@app.get("/health")
def health():

    try:
        PredictionPipeline()

        return {
            "status": "healthy",
            "model_loaded": True,
            "preprocessor_loaded": True,
            "api_version": "1.0.0"
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "preprocessor_loaded": False,
                "error": str(e)
            }
        )


# Prediction


@app.post("/predict")
def predict(data: VisaRequest):

    visa_data = USvisaData(
        continent=data.continent,
        education_of_employee=data.education_of_employee,
        has_job_experience=data.has_job_experience,
        requires_job_training=data.requires_job_training,
        no_of_employees=data.no_of_employees,
        yr_of_estab=data.yr_of_estab,
        region_of_employment=data.region_of_employment,
        prevailing_wage=data.prevailing_wage,
        unit_of_wage=data.unit_of_wage,
        full_time_position=data.full_time_position
    )

    dataframe = visa_data.get_data_as_dataframe()

    pipeline = PredictionPipeline()

    prediction, probability, feature_impacts = pipeline.predict(dataframe)

    approved_probability = float(round(float(probability[0][0]) * 100, 2))
    rejected_probability = float(round(float(probability[0][1]) * 100, 2))

    result = "Visa Approved" if prediction[0] == 0 else "Visa Rejected"

    confidence = (
        approved_probability
        if prediction[0] == 0
        else rejected_probability
    )

    return {
        "status": "success",
        "prediction": result,
        "confidence": confidence,
        "probabilities": {
            "approved": approved_probability,
            "rejected": rejected_probability
        },

        "top_features": feature_impacts
    }