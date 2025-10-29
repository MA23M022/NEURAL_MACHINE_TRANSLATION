
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from backend.schema.UserInput import userInput
from backend.schema.Model_Output import modelOutput
from backend.model.predict import m1, MODEL_VERSION, predict_output




# create end point APIs :-
app = FastAPI()

@app.get("/")
def welcome():
    return {"message" : "Welcome to Bengali to English translation app"}


@app.get("/health")
def health_check():
    return {
            "status" : "OK",
            "model_version" : MODEL_VERSION,
            "model_loaded" : m1 is not None
        }

@app.post("/predict", response_model = modelOutput)
def predict_activity(data : userInput):    
    input_lang_id = data.ben_sentence_id
    
    try:
        model_pred = predict_output(input_lang_id)
        return JSONResponse(status_code = 200,
                             content = {"ben_sentence" : model_pred["ben_sentence"], 
                                        "actual_eng_sentence" : model_pred["actual_eng_sentence"],
                                          "pred_eng_sentence" : model_pred["pred_eng_sentence"]})
    except Exception as e:
        return JSONResponse(status_code = 500, content = str(e))
    
