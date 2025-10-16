# 🌡️ Motor Temperature Prediction

This project focuses on predicting the surface temperature of a permanent magnet in a motor using various sensor readings. A Random Forest Regressor model is trained on the provided dataset to make these predictions.

---

## Dataset

The dataset used for this project is `measures_v2.csv`. It contains various sensor readings from a motor, including:

* `coolant`: Coolant temperature
* `stator_winding`: Stator winding temperature
* `u_d`, `u_q`: Voltage components
* `motor_speed`: Motor speed
* `i_d`, `i_q`: Current components
* `pm`: Permanent magnet surface temperature (target variable)

---

## Installation

To run this project, you'll need to have Python and the following libraries installed:

* **numpy**: For numerical operations
* **pandas**: For data manipulation and analysis
* **matplotlib**: For data visualization
* **seaborn**: For statistical data visualization
* **scikit-learn**: For mach
* ine learning tasks

You can install these libraries using pip:

Clone the repository:
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)

Run the Jupyter Notebook:

Open and run the mlProject.ipynb notebook to see the data exploration, model training, and evaluation steps.

Make Predictions:

The notebook also includes a function predict_temperature() that allows you to input new sensor readings and get a temperature prediction. The trained model is saved as temperature_prediction_model.pkl.

Here's an example of how to use the prediction function:

import joblib
import pandas as pd

def predict_temperature():
    """
    This function loads the trained model and predicts the permanent magnet's surface temperature
    based on user input.
    """
    model, feature_order = joblib.load('temperature_prediction_model.pkl')

    user_input = {}
    for feature in feature_order:
        value = float(input(f"Enter {feature}: "))
        user_input[feature] = value

    input_data = pd.DataFrame([user_input], columns=feature_order)

    prediction = model.predict(input_data)
    print(f"Predicted Temperature (pm): {prediction[0]}")

predict_temperature()

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
