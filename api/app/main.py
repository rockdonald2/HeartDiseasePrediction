from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.utils import Bunch
import pandas as pd
import numpy as np
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*"
]

description = '''
# Heart Disease WebAPI

Exposed endpoints used by Heart disease website.
'''

app = FastAPI(
    title="HeartDiseaseAPI",
    description=description,
    version="v0.0.1",
    contact={
        "name": "Lukács Zsolt",
        "url": "https://github.com/rockdonald2",
        "email": "lukacs.zsolt@proton.me",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

init = {}


class ClassificationParameters(BaseModel):
    bmi: float = Field(description='Body Mass Index', ge=0.0)
    smoking: bool = Field(
        description='Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]')
    alcohol_drinking: bool = Field(
        description='Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)')
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
            raise ValueError(f'Sex must be in {str(init["sex"])}.')

        return tmp

    @validator('age_category')
    def check_age_category(cls, v):
        if v not in init['age_category']:
            raise ValueError(
                f'Age category must be in {str(init["age_category"])}.')

        return v

    @validator('race')
    def check_race(cls, v):
        if v not in init['race']:
            raise ValueError(f'Race must be in {str(init["race"])}.')

        return v

    @validator('general_health')
    def check_gen_health(cls, v):
        if v not in init['gen_health']:
            raise ValueError(
                f'General health must be in {str(init["gen_health"])}.')

        return v


class ClassificationPragmaticalBarebone(BaseModel):
    '''
        Used internally only for prediction model.
    '''

    bmi: Optional[float]
    smoking: Optional[int] = Field(ge=0, le=1)
    alcohol_drinking: Optional[int] = Field(ge=0, le=1)
    stroke: Optional[int] = Field(ge=0, le=1)
    physical_health: Optional[int] = Field(ge=0, le=30)
    mental_health: Optional[int] = Field(ge=0, le=30)
    diff_walking: Optional[int] = Field(ge=0, le=1)
    sex: Optional[int] = Field(ge=0, le=1)
    age_category: Optional[int]
    race: Optional[int]
    diabetic: Optional[int] = Field(ge=0, le=1)
    physical_activity: Optional[int]
    general_health: Optional[int]
    sleep_time: Optional[float]
    asthma: Optional[int] = Field(ge=0, le=1)
    kidney_disease: Optional[int] = Field(ge=0, le=1)
    skin_cancer: Optional[int] = Field(ge=0, le=1)

    def to_list(self) -> list:
        return [self.bmi, self.smoking, self.alcohol_drinking, self.stroke, self.physical_health, self.mental_health, self.diff_walking, self.sex, self.age_category, self.race, self.diabetic, self.physical_activity, self.general_health, self.sleep_time, self.asthma, self.kidney_disease, self.skin_cancer]

class ClassificationEncodedBarebone(BaseModel):
    '''
        Used internally only for prediction model.
    '''

    bmi: Optional[float]
    smoking: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    alcohol_drinking: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    stroke: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    physical_health: Optional[int] = Field(ge=0, le=30)
    mental_health: Optional[int] = Field(ge=0, le=30)
    diff_walking: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    sex: Optional[str] = Field(regex=r'(^Male$)|(^Female$)')
    age_category: Optional[str]
    race: Optional[str]
    diabetic: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    physical_activity: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    general_health: Optional[int]
    sleep_time: Optional[float]
    asthma: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    kidney_disease: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')
    skin_cancer: Optional[str] = Field(regex=r'(^No$)|(^Yes$)')

    def to_list(self) -> list:
        return [self.bmi, self.physical_health, self.mental_health, self.sleep_time, self.smoking, self.alcohol_drinking, self.stroke,  self.diff_walking, self.sex, self.age_category, self.race, self.diabetic, self.physical_activity, self.general_health, self.asthma, self.kidney_disease, self.skin_cancer]

class ClassificationResult(BaseModel):
    no: float = Field(description="Probability that heart disease won't occur")
    yes: float = Field(description="Probability that heart disease will occur")


def populate_dropdowns(uncleaned: pd.DataFrame):
    init['sex'] = uncleaned['Sex'].unique().tolist()
    init['sex'].sort()
    init['age_category'] = uncleaned['AgeCategory'].unique().tolist()
    init['age_category'].sort()
    init['race'] = uncleaned['Race'].unique().tolist()
    init['race'].sort()
    init['gen_health'] = uncleaned['GenHealth'].unique().tolist()
    init['gen_health'].sort()


def prepare_prediction_model1_with_pragmatical_conversion():
    '''
        Populates init dictionary with sex, age_category, race, gen_health and model keys.
        Using manual categorical-numerical conversion.
    '''

    heart = pd.read_csv('data/heart_cleaned.csv')

    heart_model_data = Bunch()
    heart_model_data['target'] = heart['HeartDisease'].to_list()
    heart.drop(columns=['HeartDisease'], inplace=True)
    heart_model_data['data'] = heart.values.tolist()
    heart_model_data['feature_names'] = heart.columns.to_list()
    heart_model_data['target_names'] = ['No', 'Yes']

    del heart

    heart_uncleaned = pd.read_csv('data/heart_2020_cleaned.csv')
    populate_dropdowns(heart_uncleaned)
    del heart_uncleaned

    X = heart_model_data.data
    y = heart_model_data.target

    init['model'] = make_pipeline(StandardScaler(), SGDClassifier(
        loss='modified_huber', penalty='l2', max_iter=1000, class_weight='balanced', random_state=0))

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=2000)

    init['model'].fit(X_train, y_train)


def prepare_prediction_model2_with_encoders():
    '''
        Populates init dictionary with sex, age_category, race, gen_health, features and model keys.
        Using sci-kit's Label and Ordinal encoders.
        Also populates init dict with encoders.
    '''

    heart = pd.read_csv('data/heart_2020_cleaned.csv')
    populate_dropdowns(heart)

    y = heart['HeartDisease']
    heart.drop(columns=['HeartDisease'], inplace=True)

    le = LabelEncoder()
    le.fit(y)
    fitted_y = le.transform(y)

    heart_model_data = Bunch()

    heart_model_data['target_names'] = le.classes_
    heart_model_data['target'] = fitted_y

    X_numerical = heart[['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']]
    X_categorical = heart.drop(columns=['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime'])

    oe = OrdinalEncoder()
    oe.fit(X_categorical)
    fitted_X_categorical = oe.transform(X_categorical)

    X_numerical.reset_index(inplace=True)

    fitted_X_categorical = pd.DataFrame(fitted_X_categorical, columns=X_categorical.columns)
    fitted_X_categorical.reset_index(inplace=True)

    X = X_numerical.merge(fitted_X_categorical, left_on='index', right_on='index', how='inner')
    X.drop(columns='index', inplace=True)

    heart_model_data['data'] = X.values
    heart_model_data['feature_names'] = X.columns

    init['features'] = heart_model_data['feature_names']

    init['encoder_y'] = le
    init['encoder_X'] = oe

    init['model'] = make_pipeline(StandardScaler(), SGDClassifier(
        loss='log', penalty='l2', max_iter=1000, class_weight='balanced', random_state=0))

    X = heart_model_data.data
    y = heart_model_data.target

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=2000)

    init['model'].fit(X_train, y_train)


def do_pragmatical_prediction(input: ClassificationParameters):
    data = ClassificationPragmaticalBarebone()

    data.bmi = input.bmi
    data.smoking = int(input.smoking)
    data.alcohol_drinking = int(input.alcohol_drinking)
    data.stroke = int(input.stroke)
    data.physical_health = input.physical_health
    data.mental_health = input.mental_health
    data.diff_walking = int(input.diff_walking)
    data.sex = init['sex'].index(input.sex)
    data.age_category = init['age_category'].index(input.age_category)
    data.race = init['race'].index(input.race)
    data.diabetic = int(input.diabetic)
    data.physical_activity = int(input.physical_activity)
    data.general_health = init['gen_health'].index(input.general_health)
    data.sleep_time = input.sleep_time
    data.asthma = int(input.asthma)
    data.kidney_disease = int(input.kidney_disease)
    data.skin_cancer = int(input.skin_cancer)

    result = init['model'].predict_proba([data.to_list()])[0]
    return ClassificationResult(no=result[0], yes=result[1])


def do_encoded_prediction(input: ClassificationParameters):
    data = ClassificationEncodedBarebone()

    data.bmi = input.bmi
    data.smoking = 'Yes' if input.smoking else 'No'
    data.alcohol_drinking = 'Yes' if input.alcohol_drinking else 'No'
    data.stroke = 'Yes' if input.stroke else 'No'
    data.physical_health = input.physical_health
    data.mental_health = input.mental_health
    data.sex = input.sex
    data.age_category = input.age_category
    data.race = input.race
    data.diff_walking = 'Yes' if input.diff_walking else 'No'
    data.diabetic = 'Yes' if input.diabetic else 'No'
    data.physical_activity = 'Yes' if input.physical_activity else 'No'
    data.general_health = input.general_health
    data.sleep_time = input.sleep_time
    data.asthma = 'Yes' if input.asthma else 'No'
    data.kidney_disease = 'Yes' if input.kidney_disease else 'No'
    data.skin_cancer = 'Yes' if input.skin_cancer else 'No'

    data_df = pd.DataFrame([data.to_list()], columns=init['features'])

    df_numerical = data_df[init['features'][:4]]
    df_categorical = data_df[init['features'][4:]]

    data_encoded = df_numerical.values.tolist()
    data_encoded[0].extend(init['encoder_X'].transform(df_categorical.to_numpy().reshape(1, -1))[0])

    result = init['model'].predict_proba(data_encoded)[0]
    return ClassificationResult(no=result[0], yes=result[1])

@app.on_event('startup')
def on_startup():
    '''
        Create classification model.
    '''

    # prepare_prediction_model1_with_pragmatical_conversion()
    prepare_prediction_model2_with_encoders()


@app.post('/api/classify', response_model=ClassificationResult, tags=['API'])
def do_classify(input: ClassificationParameters):
    '''
        We will need to convert the input parameters to be fitting to our model.
    '''

    # return do_pragmatical_prediction(input)
    return do_encoded_prediction(input)


@app.get('/api/get/sex', tags=['API'])
def get_sex():
    if 'sex' in init:
        return {
            'sex': init['sex']
        }
    else:
        raise HTTPException(status_code=404, detail='Not found')


@app.get('/api/get/age', tags=['API'])
def get_age():
    if 'age_category' in init:
        return {
            'age_category': init['age_category']
        }
    else:
        raise HTTPException(status_code=404, detail='Not found')


@app.get('/api/get/race', tags=['API'])
def get_race():
    if 'race' in init:
        return {
            'race': init['race']
        }
    else:
        raise HTTPException(status_code=404, detail='Not found')


@app.get('/api/get/gen_health', tags=['API'])
def get_gen_health():
    if 'gen_health' in init:
        return {
            'gen_health': init['gen_health']
        }
    else:
        raise HTTPException(status_code=404, detail='Not found')
