from fastapi import FastAPI
import joblib
import uvicorn 
from schemas import Characteristic


app = FastAPI()
# Load the previously trained model
model = joblib.load('iris_model.pkl')
model_labels = ['setosa', 'versicolor', 'virginica']


@app.post('/predict')
def predict(iris:Characteristic):
    """ This function using method post the values sepal_lenght, sepal_width, petal_length and petal_width in the endpoint predict 

    Args:
        iris (Class): The class defined in the module schemas.py, this class contains the parameters sepal_lenght, sepal_width, petal_length and petal_width.

    Returns:
        dict: The prediction of the model with the labels previously defined.
    """
    data = iris.dict()
    features = [[ data['sepal_length'],
                 data['sepal_width'],
                 data['petal_length'],
                 data['petal_width'] ]]
    prediction = model.predict(features)
    result = model_labels[prediction[0]]
    return {'prediction': result }

if __name__ == '__main__':
    """
     Run the server in the url http://localhost:5000.
     If you go to http://localhost:5000/docs you fin the documentation.
    
    Example
    --------
    curl -X 'POST' \
  'http://localhost:5000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.0,
  "sepal_width": 2.4,
  "petal_length": 1.2,
  "petal_width": 1
   }'
   Response:
   --------
   {"prediction": "setosa"}
    """
    uvicorn.run('main:app', host = 'localhost', port=5000, reload = True)