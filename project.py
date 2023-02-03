import matplotlib.pylab as plt
import wfdb
import matplotlib.pyplot as plt
import pywt
import numpy as np
from wfdb import processing
import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
import xgboost as xg
import pandas as pd
from sklearn import svm

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

#extract features
def extract_udf(_df):
  extraction_settings = ComprehensiveFCParameters()
  X = extract_features(_df, column_id='id', column_sort='time', default_fc_parameters=extraction_settings, impute_function=impute)
  return X

def split_data(tc_x, tc_y):
  train_X, test_X, train_y, test_y = train_test_split(tc_x, tc_y, test_size = 0.2, random_state = 121)
  return train_X, test_X, train_y, test_y

def load_data_create_model():
    # Load Data
    data_dir = './data'
    resps_hr = []
    for infant_num in range(1, 11):
        hr_all, resp_all = [], []
        for line in open(f"{data_dir}/infant{infant_num:d}_ori_sample2.txt"):
            f = line.strip().split('\t')
            if len(f) != 2: continue
            hr, resp = f[0], eval(f[1])
            if hr == "nan": continue
            hr_all.append(float(hr))
            resp_all.append(resp[0])

        hr_std = np.std([rd for rd in hr_all if rd > 80])
        hr_mean = np.mean([rd for rd in hr_all if rd > 80])
        hr_min_thr = hr_mean - 3 * hr_std
        hr_max_thr = hr_mean + 3 * hr_std
        print(hr_std, hr_mean, hr_min_thr, hr_max_thr)

        resp_std = np.std([rd for rd in resp_all if rd > 10])
        resp_mean = np.mean([rd for rd in resp_all if rd > 10])
        resp_min_thr = resp_mean - 4 * resp_std
        resp_max_thr = resp_mean + 4 * resp_std
        print(resp_std, resp_mean, resp_min_thr, resp_max_thr)

        # ----------------------------------------------------------

        resp_hr = []
        for m in range(len(hr_all)):
            if hr_min_thr < hr_all[m] < hr_max_thr:
                # resp_hr.append([hr_all[m], resp_all[m]])
                if resp_min_thr < resp_all[m] < resp_max_thr:
                    resp_hr.append([hr_all[m], resp_all[m]])
                # if len(resp_hr) == 3000000: break
        print(len(resp_hr))

        # ----------------------------------------------------------

        base_hr = 0
        empty_resp = []
        for item in resp_hr[20000:200000]:
            _hr, _resp = item
            if _hr != base_hr:
                if empty_resp:
                    resps_hr.append([base_hr, empty_resp])
                empty_resp = []
                base_hr = _hr
                empty_resp.append(_resp)
            else:
                empty_resp.append(_resp)

        resps_hr.append([base_hr, empty_resp])
        print(len(resps_hr))
        print(resps_hr[0])
        # ----------------------------------------------------------

    #Indicate the position of each infant in the sample
    infant_content = {"infant1": [0, 5667], "infant2": [5667, 12345], "infant3": [12345, 18627],
                      "infant4": [18627, 25797], "infant5": [25797, 27451], "infant6": [27451, 33505],
                      "infant7": [33505, 40923], "infant8": [40923, 47429], "infant9": [47429, 53657],
                      "infant10": [53657, 61083]}

    #convert to DataFrame
    hr_y_list, resp_x_list = [], [[], []]
    startid = 0
    filename = ""
    for _index in range(len(resps_hr)):
        _hr, _resps_list = resps_hr[_index]
        if len(_resps_list) < 27: continue
        hr_y_list.append(_hr)
        if infant_content["infant1"][0] <= _index < infant_content["infant1"][1]:
            filename = "infant1_"
        elif infant_content["infant2"][0] <= _index < infant_content["infant2"][1]:
            filename = "infant2_"
        elif infant_content["infant3"][0] <= _index < infant_content["infant3"][1]:
            filename = "infant3_"
        elif infant_content["infant4"][0] <= _index < infant_content["infant4"][1]:
            filename = "infant4_"
        elif infant_content["infant5"][0] <= _index < infant_content["infant5"][1]:
            filename = "infant5_"
        elif infant_content["infant6"][0] <= _index < infant_content["infant6"][1]:
            filename = "infant6_"
        elif infant_content["infant7"][0] <= _index < infant_content["infant7"][1]:
            filename = "infant7_"
        elif infant_content["infant8"][0] <= _index < infant_content["infant8"][1]:
            filename = "infant8_"
        elif infant_content["infant9"][0] <= _index < infant_content["infant9"][1]:
            filename = "infant9_"
        elif infant_content["infant10"][0] <= _index < infant_content["infant10"][1]:
            filename = "infant10_"
        resp_x_list[0].extend(_resps_list)
        resp_x_list[1].extend([filename + str(startid)] * len(_resps_list))
        startid += 1

    c = {"resp": resp_x_list[0], "id": resp_x_list[1], "time": [i for i in range(len(resp_x_list[0]))]}
    dfx = pd.DataFrame(c)

    #extract features, output is n*783
    X = extract_udf(dfx)

    dfid = pd.DataFrame(dfx["id"])
    dfid.drop_duplicates("id", keep='first', inplace=True)

    dfid["y"] = hr_y_list
    print(dfid.shape)

    extend_fea = []
    for _id in dfid["id"]:
        extend_fea.append((X.loc[_id].values))

    for i in range(783):
        _feas = []
        for item in extend_fea:
            _feas.append(item[i])
        dfid["extend_fea_" + str(i)] = _feas

    # use infant4 and infant5 as evaluation, others as train to create model, use mse and r2 to evaluate
    for i in range(4, 10):
        print(i)
        a = [i, i + 1]
        b = [m for m in range(1, 11) if m not in a]
        a.extend(b)

        va = dfid[dfid.id.str.contains('^infant{0}_'.format(a[0]))]
        vb = dfid[dfid.id.str.contains('^infant{0}_'.format(a[1]))]

        vc_x = np.concatenate((va.iloc[:, 2:].values, vb.iloc[:, 2:].values))
        vc_y = np.concatenate((va.iloc[:, 1].values, vb.iloc[:, 1].values))

        ta = dfid[dfid.id.str.contains('^infant{0}_'.format(a[2]))]
        tb = dfid[dfid.id.str.contains('^infant{0}_'.format(a[3]))]
        tc = dfid[dfid.id.str.contains('^infant{0}_'.format(a[4]))]
        td = dfid[dfid.id.str.contains('^infant{0}_'.format(a[5]))]
        te = dfid[dfid.id.str.contains('^infant{0}_'.format(a[6]))]
        tf = dfid[dfid.id.str.contains('^infant{0}_'.format(a[7]))]
        tg = dfid[dfid.id.str.contains('^infant{0}_'.format(a[8]))]
        th = dfid[dfid.id.str.contains('^infant{0}_'.format(a[9]))]

        tc_x = np.concatenate((ta.iloc[:, 2:].values, tb.iloc[:, 2:].values, tc.iloc[:, 2:].values,
                               td.iloc[:, 2:].values, te.iloc[:, 2:].values, tf.iloc[:, 2:].values,
                               tg.iloc[:, 2:].values, th.iloc[:, 2:].values))
        tc_y = np.concatenate((ta.iloc[:, 1].values, tb.iloc[:, 1].values, tc.iloc[:, 1].values, td.iloc[:, 1].values,
                               te.iloc[:, 1].values, tf.iloc[:, 1].values, tg.iloc[:, 1].values, th.iloc[:, 1].values))

        print(tc_x.shape, tc_y.shape, vc_x.shape, vc_y.shape)

        train_X, test_X, train_y, test_y = split_data(tc_x, tc_y)
        print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

        xgb_r = xg.XGBRegressor(objective='reg:squarederror', n_estimators=300, seed=123)

        # Fitting the model
        xgb_r.fit(train_X, train_y)

        # test
        pred = xgb_r.predict(test_X)
        tmse = np.sqrt(MSE(test_y, pred))
        tr2 = r2_score(test_y, pred)

        # val
        pred = xgb_r.predict(vc_x)
        vmse = np.sqrt(MSE(vc_y, pred))
        vr2 = r2_score(vc_y, pred)

        print(str(i) + '\t' + str(tmse) + '\t' + str(tr2) + '\t' + str(vmse) + '\t' + str(vr2))

        plt.figure(figsize=(15.0, 3.0))
        plt.ylim((60, 200))
        plt.title("HR", size=20, color="blue")
        plt.plot(pred, label='real Hr')
        plt.plot(vc_y, label='predict Hr')
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

        break


if __name__ == "__main__":
    load_data_create_model()





