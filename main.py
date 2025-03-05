import uvicorn
from fastapi import FastAPI
from inference import get_predictions
from loguru import logger
from pydantic import BaseModel

title = "CNN Model API"
description = "A simple API to load and predict with CNN model in fast API"

# Initiate app instance
app = FastAPI(title=title, version="1.0", description=description)


class IncomingData(BaseModel):
    queries: list


# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(
    sink="app/data/log_files/logs.log",
    format=log_format,
    level="DEBUG",
    compression="zip",
)


# Api root or home endpoint
@app.get("/")
@app.get("/health")
def read_home():
    """
    Health endpoint which can be used to test the availability of the application.
    :return: Dict with key 'message' and value
    """
    logger.debug("User checked the root page")
    return {"message": f"{title} - live!"}


# Prediction endpoint
@app.post("/predict")
@logger.catch()  # catch any unexpected breaks
def get_predictions_from_model(incoming_data: IncomingData):
    """
    Prediction endpoint to process the raw queries and pass them to model for inferencing and return the results
    :return List with predicted category and props
    """
    data = incoming_data.model_dump()
    if data["queries"]:
        queries_parsed = data["queries"]
        logger.info(f"User sent queries for predictions are: {queries_parsed}")

        preds = get_predictions(queries_parsed)
        logger.debug("Predictions successfully generated for the user")

        return preds
    return "No queries found"


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")
