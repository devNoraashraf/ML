import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS  # ✅ استيراد CORS

app = Flask(__name__)
CORS(app)  # ✅ تفعيل CORS

model = pickle.load(open("model.pkl", "rb"))

@app.route('/', methods=['POST'])
def process_array():
    data = request.get_json()

    if 'array' not in data:
        return jsonify({"error": "No array key found in JSON"}), 400

    array_data = data['array']
    print(len(array_data))
    features = np.array([array_data])
    predicted_labels = model.predict(features)

    data = pd.read_csv("medical_data.csv").iloc[:, :-2]
    features_name = list(data['spicitalist'])
    features_name.append('spicitalist')
    class_names = data.columns[1:]
    data_reshaped = data.drop(['spicitalist'], axis=1)
    data_reshaped.loc[len(data_reshaped.index)] = class_names

    data_reshaped = pd.DataFrame(data_reshaped.values.T, columns=features_name)

    X = data_reshaped.drop(['spicitalist'], axis=1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(data_reshaped['spicitalist'])

    print("Predicted Labels:", predicted_labels)

    predicted_labels_names = encoder.inverse_transform(predicted_labels)
    print("Predicted Labels Names:", predicted_labels_names)

    result = predicted_labels_names.tolist()
    return json.dumps(result)

if __name__ == "__main__":
    app.run()
