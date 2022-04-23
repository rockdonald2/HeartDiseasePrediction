from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError, validator
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch
import pandas as pd
import numpy as np


app = FastAPI()

init = {}

class ClassificationParameters(BaseModel):
    bmi: float = Field(description='Body Mass Index', ge=0.0)
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

    @validator('sex')
    def check_sex(cls, v):
        tmp = v[0].upper() + v[1:].lower()

        if tmp not in init['sex']:
            raise ValidationError(f'Sex must be in {str(init["sex"])}.')

        return tmp

    @validator('age_category')
    def check_age_category(cls, v):
        if v not in init['age_category']:
            raise ValidationError(f'Age category must be in {str(init["age_category"])}.')
        
        return v

    @validator('race')
    def check_race(cls, v):
        if v not in init['race']:
            raise ValidationError(f'Race must be in {str(init["race"])}.')

        return v

    @validator('general_health')
    def check_gen_health(cls, v):
        if v not in init['gen_health']:
            raise ValidationError(f'General health must be in {str(init["gen_health"])}.')
        
        return v


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

    heart_uncleaned = pd.read_csv('../data/heart_2020_cleaned.csv')
    init['sex'] = heart_uncleaned['Sex'].unique().tolist()
    init['age_category'] = heart_uncleaned['AgeCategory'].unique().tolist()
    init['race'] = heart_uncleaned['Race'].unique().tolist()
    init['gen_health'] = heart_uncleaned['GenHealth'].unique().tolist()

    del heart_uncleaned

    X = heart_model_data.data
    y = heart_model_data.target

    init['sgd'] = make_pipeline(StandardScaler(), SGDClassifier(loss='modified_huber', penalty='elasticnet', tol=1e-6, max_iter=np.ceil(10**8 / len(heart_model_data.data))))
    init['sgd'].fit(X, y)


@app.post('/api/classify')
def do_classify(input: ClassificationParameters):
    '''
        We will need to convert the input parameters to be fitting to our model.
    '''

    pass

@app.get('/api/get/sex')
def get_sex():
    return {
        'sex': init['sex']
    }

@app.get('/api/get/age')
def get_age():
    return {
        'age_category': init['age_category']
    }

@app.get('/api/get/race')
def get_race():
    return {
        'race': init['race']
    }

@app.get('/api/get/gen_health')
def get_gen_health():
    return {
        'gen_health': init['gen_health']
    }