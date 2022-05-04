# README

Machine learning based algorithm to diagnose heart disease.

## How to get it to work

1. `git clone` this repo,
2. build `web` project with `cd web/app && npm run build && cd ../..`
3. `docker compose up -d` to install services,
4. wait a minute, then ready to go.

You have a container deployed Jupyter Lab (port `8889`), FastAPI with uvicorn (port `1111`) and NGINX (port `80`).

## Tests

We tried these supervised-classification learning methods:

- `KNeighbors` with multiple n-neighbors values: was totally useless, with results that: n-neighbors value of 1 or 2 was the most precise, which is unreal;
- `SVC` good score, really bad predicted probabilities;
- `SGD` had good results, with averages scores around 70-75%, but this depends on a lot factors discussed in the section [Data processing](#data-processing);
- `LogisticRegression` same as SGD, really hard to tell which is better, depends on how the data is formulated for processing.

## Data processing

The real challenge is that somehow we need to use the aforementioned algorithms with categorical data.

There are multiple ways by which we can convert categorical columns into numerical columns:  

- we can assign programatically numbers to categorical values, i.e. sort the unique values of a category, then take this list's indices and assign the its index to a value.
  - as it turns out this is also a supported encoding method in sci-kit named `LabelEncoder` or `OrdinalEncoder` (see here: [source](https://www.ritchieng.com/machinelearning-one-hot-encoding/)).
- we can use `pd.get_dummies()` that does this in a different manner;
- we can use `OneHotEncoder` from sci-kit.

We different data forms we had different scores, but the average score was somewhere between 70-75%.

With the first data preparation method the `SGDClassifier` was the obvious winner, but if we used the `pd.get_dummies()` method `LogisticRegression` came up first.

### First try

| Classifier | random `.score()` value when using first preparation method |
| --- | --- |
| `SGDClassifier` with `modified_huber` and `class_weight='balanced'` | 0.7583489681050657 |
| `SGDClassifier` with `log` and `class_weight='balanced'` | 0.743370856785491 |
| `LogisticRegression` with `class_weight='balanced'` | 0.7429643527204502 |

_All the results can be inspected in the `explore.ipynb` notebook._

### Second try

| Classifier | random `.score()` value when using `get_dummies()` preparation method |
| --- | --- |
| `SGDClassifier` with `modified_huber` and `class_weight='balanced'` | 0.7189493433395873 |
| `SGDClassifier` with `log` and `class_weight='balanced'` | 0.7651657285803627 |
| `LogisticRegression` with `class_weight='balanced'` | 0.7502188868042526 |

With `SGDClassifier` and `log` we had awful predictions, even if `modified_huber` had the lowest score it predicted the best.

_All the results can be inspected in the `explore_dummies.ipynb` notebook._

### Third try

| Classifier | random `.score()` value when using `OneHotEncoder` preparation method |
| --- | --- |
| `SGDClassifier` with `modified_huber` and `class_weight='balanced'` | 0.7271106941838649 |
| `SGDClassifier` with `log` and `class_weight='balanced'` | 0.724015009380863 |
| `LogisticRegression` with `class_weight='balanced'` | 0.7537836147592245 |

_All the results can be inspected in the `explore_hotencoder.ipynb` notebook._
_Could not test it, took too long to predict, but with seemingly good precision._

### Fourth try

| Classifier | random `.score()` value when using `LabelEncoder` and `OrdinalEncoder` |
| --- | --- |
| `SGDClassifier` with `modified_huber` and `class_weight='balanced'` | 0.7583489681050657. |
| `SGDClassifier` with `log` and `class_weight='balanced'` | 0.743370856785491 |
| `LogisticRegression` with `class_weight='balanced'` | 0.7429643527204502 |

_All the results can be inspected in the `explore_label_ordinal_encoders.ipynb` notebook._
Bit of hack made here, we separated numerical and categorical columns, then applied OrdinalEncoder on categorical values.

## Implementation

`web` project contains the front-end, written in HTML + SCSS + JS (webpack).
`api` project contains the back-end, written in Python.

Every Docker deployment will deploy the `web` and `api` projects in separate containers.

In the `api` project we have implemented two ways of predicting results:

- one is the manual, pragmatical method, where we manually assign integer values to categories, with: `prepare_prediction_model1_with_pragmatical_conversion()` and `return do_pragmatical_prediction(input)` calls;
- second uses the method with `OrdinalEncoder` and `LabelEncoder`, with `prepare_prediction_model2_with_encoders()` and `return do_encoded_prediction(input)` calls.

Current deployment contains the second approach.

---

### Other remarks

- Before trying to use the api in the website, update the API address in the `consts.js` file;
- The API currently uses the first discussed method for categorical-numerical conversion and SGDClassifier to predict with good precision.
- We've also used classification_reports to measure the accurracy of our models.
