import os

import numpy as np
import sklearn.metrics as sm

from src.ucr_data.load_ucr_pre import load_ucr

# from sklearn.preprocessing import LabelEncoder
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance


if __name__ == '__main__':
    print("Hello world!!!")
    ucr = ['ArrowHead', 'CBF', 'ECG200', 'ECG5000', 'ECGFiveDays', 'FaceFour', 'GunPoint', 'InsectWingbeatSound',
           'MedicalImages', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OSULeaf', 'Plane',
           'Symbols', 'Trace', 'TwoLeadECG']
    uea = ['FaceDetection', 'FingerMovements', 'HandMovementDirection',
           'Handwriting', 'Heartbeat', 'Libras', 'MotorImagery', 'NATOPS', 'PEMS-SF', 'RacketSports',
           'SelfRegulationSCP1', 'UWaveGestureLibrary']
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    print("father_path = ", father_path)
    print("uea_len = ", len(uea), ", ucr_len = ", len(ucr))
    result_value = []
    for dataset_name in ucr:
        # dataset_name = 'CBF'
        print("dataset_name = ", dataset_name)
        X, y = load_ucr(dataset_name)
        print(X.shape, y.shape)
        print("X.shape = ", X.shape)
        X1 = np.reshape(X, (X.shape[0], -1))
        print("X1.shape = ", X1.shape)
        all_sc = sm.silhouette_score(X1, y)
        result_evalution = dict()
        result_evalution['dataset_name'] = dataset_name
        result_evalution['raw_sc'] = round(all_sc, 4)
        result_value.append(result_evalution)
        # path = father_path + '/sc_results/' + '16_ucr_sc_20211227.csv'
        # pd.DataFrame(result_value).to_csv(path)