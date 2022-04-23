from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch
import pandas as pd
import numpy as np


app = FastAPI()

init = {}

class ClassificationParameters(BaseModel):
    bmi: float = Field(description='Body Mass Index')
    smoking: bool = Field(
        description='Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]')
    alcohol_drinking: bool = Field(
        description='Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week')
    stroke: bool = Field(description='(Ever told) (you had) a stroke?')
    physical_health: int = Field(
        description='Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days)',  ge=0, le=30)
    mental_health: int = Field(
        description='Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days)', ge=0, le=30)
    diff_walking: bool = Field(
        description='Do you have serious difficulty walking or climbing stairs?')
    sex: str = Field(description='Are you male or female?')
    age_category: str = Field(description='Fourteen-level age category')
    race: str = Field(description='Imputed race/ethnicity value')
    diabetic: bool = Field(description='(Ever told) (you had) diabetes?')
    physical_activity: bool = Field(
        description='Adults who reported doing physical activity or exercise during the past 30 days other than their regular job')
    general_health: str = Field(
        description='Would you say that in general your health is...')
    sleep_time: float = Field(
        description='On average, how many hours of sleep do you get in a 24-hour period?')
    asthma: bool = Field(description='(Ever told) (you had) asthma?')
    kidney_disease: bool = Field(
        description='Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?')
    skin_cancer: bool = Field(description='(Ever told) (you had) skin cancer?')


@app.on_event('startup')
def on_startup():
    '''
        Create classification model.
    '''
    heart = pd.read_csv('../data/heart_cleaned.csv')

    heart_model_data = Bunch()
    heart_model_data['target'] = heart['HeartDisease'].to_list()
    heart.drop(columns=['HeartDisease'], inplace=True)
    heart_model_data['data'] = heart.values.tolist()
    heart_model_data['feature_names'] = heart.columns.to_list()
    heart_model_data['target_names'] = ['No', 'Yes']

    del heart

    X = heart_model_data.data
    y = heart_model_data.target

    init['sgd'] = make_pipeline(StandardScaler(), SGDClassifier(loss='modified_huber', penalty='elasticnet', tol=1e-6, max_iter=np.ceil(10**8 / len(heart_model_data.data))))
    init['sgd'].fit(X, y)


@app.post('/api/classify')
def do_classify():
    pass
