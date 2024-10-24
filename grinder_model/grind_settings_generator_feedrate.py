import pathlib
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import math

from volume_predictor_svr import load_model, load_scaler


def volume_mismatch_penalty(x, volume, wear, model, scaler, rpm):
    predicted_volume = predict_volume(x[0], rpm, x[1], wear, model, scaler)
    return (volume - predicted_volume) ** 2


def predict_volume(force, rpm, time, wear, model, scaler):
    input_data_dict = {
                       'grind_time': [time],
                       'avg_rpm': [rpm],
                       'avg_force': [force],
                       'initial_wear': [wear]
    }
    input_df = pd.DataFrame(input_data_dict)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    predicted_volume = model.predict(input_scaled)[0][0]
    return predicted_volume


def generate_settings(volume, wear, model, scaler, rpm_model, rpm_scaler, rpm=11000):
    
    # x = [force, time]
    x = [5, 10]
    min_f, max_f = 3, 9
    min_t, max_t = 5, 20
    result = minimize(volume_mismatch_penalty, x, args=(volume, wear, model, scaler, rpm),
                      bounds=((min_f, max_f), (min_t, max_t)))

    input_rpm_correction_data_dict = {
        'avg_force': [result.x[0]],
        'rpm_setpoint': [rpm]
    }
    input_df = pd.DataFrame(input_rpm_correction_data_dict)
    input_scaled = rpm_scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
    predicted_avg_rpm = rpm_model.predict(input_scaled)


    settings = {
                'force': result.x[0],
                'time': result.x[1],
                'rpm': predicted_avg_rpm,
    }
    predicted_volume = predict_volume(settings['force'], settings['rpm'], settings['time'], wear, model, scaler)
    
    if vol > 130:
        mrr = predicted_volume / settings['time']
        settings['time'] = volume / mrr
        predicted_volume = volume


    return settings, predicted_volume



if __name__ == '__main__':

    rpm_correction_model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'rpm_correction_model_svr_V1.pkl'
    rpm_correction_scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'rpm_correction_scaler_svr_V1.pkl'
    rpm_correction_model = load_model(use_fixed_path=True, fixed_path=rpm_correction_model_path)
    rpm_correction_scaler = load_scaler(use_fixed_path=True, fixed_path=rpm_correction_scaler_path)

    model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_V1.pkl'
    scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_V1.pkl'

    grind_model = load_model(use_fixed_path=True, fixed_path=model_path)
    grind_scaler = load_scaler(use_fixed_path=True, fixed_path=scaler_path)

    removed_material = np.arange(80, 200, 10)
    wear_range = np.linspace(1e6, 3e6, 2)
    belt_width = 0.025                          #in m 
    belt_angle = 30                             #in degree
    contact_width = belt_width * math.sin(math.radians(belt_angle))

    for vol in removed_material:
        for wear in wear_range:
            grind_settings, predicted_volume_loss = generate_settings(vol, wear, grind_model, grind_scaler, rpm_correction_model, rpm_correction_scaler, 10000)
            feed_rate = belt_width / grind_settings["time"]
            print(f'\n\nSettings:\n  force: {grind_settings["force"]}\n  rpm:{grind_settings["rpm"]}\n  time: {grind_settings["time"]}\n  Feed_rate: {feed_rate * 1000} mm/s')
            print(f'Removed material\n  input: {vol}\n  predicted: {predicted_volume_loss}')





