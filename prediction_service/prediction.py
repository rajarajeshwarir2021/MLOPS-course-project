import yaml
import joblib
import numpy as np

params_path = "params.yaml"

def read_params(config_path):
    """
    Read the parameters
    """
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def predict(data):
    """
    Predict the target for the given parameters in list format
    """
    config = read_params(params_path)
    model_dir_path = config['webapp_model_dir']
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    print(prediction)
    return prediction[0]


def api_response(request):
    """
    Predict the target for the given parameters in json format
    """
    try:
        data = np.array([list(request.json.values())])
        response = predict(data=data)
        response = {"response": response}
        return response
    except Exception as e:
        print(e)
        error_message = "Error: " + str(e)
        return error_message
