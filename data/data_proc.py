import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter  # 🚀 引入 SG 滤波器

TARGET_DT = 2e-3
SG_WINDOW = 51  # 500Hz ~ 0.7 s


def pack_csv_to_pickle(train_dir, test_dir, output_filename="character_data.pickle"):
    data = {
        "labels": [], "t": [], "qp": [], "qv": [], "qa": [], "tau": [],
        "m": [], "c": [], "g": [], "p": [], "pdot": []
    }

    folder_configs = [
        (train_dir, "train_labels"),
        (test_dir, "test_labels")
    ]

    for folder_path, label_str in folder_configs:
        if not os.path.exists(folder_path):
            print(f"folder '{folder_path}' unexists, skipping...")
            continue

        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
        print(f" '{folder_path}':{len(csv_files)} files...")

        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            try:
                df = pd.read_csv(file_path)

                df = df.groupby("time").mean().reset_index()
                num_samples = len(df)


                raw_time = df['time'].values
                qp_arr = df[['pos_meas1', 'pos_meas2']].values
                qv_arr = df[['vel_meas1', 'vel_meas2']].values
                tau_arr = df[['tau_meas1', 'tau_meas2']].values


                t_arr = raw_time[0] + np.arange(num_samples) * TARGET_DT

                window = SG_WINDOW
                if num_samples <= window:
                    window = num_samples - 1 if num_samples % 2 != 0 else num_samples - 2

                if window >= 5:
                    qv_arr = np.stack([savgol_filter(qv_arr[:, i], window, 3) for i in range(2)], axis=1)
                    tau_arr = np.stack([savgol_filter(tau_arr[:, i], window, 3) for i in range(2)], axis=1)

                qa_arr = np.gradient(qv_arr, t_arr, axis=0)

                m_arr = np.zeros((num_samples, 2))
                c_arr = np.zeros((num_samples, 2))
                g_arr = np.zeros((num_samples, 2))
                p_arr = np.zeros((num_samples, 2))
                pdot_arr = np.zeros((num_samples, 2))

                data["labels"].append(label_str)
                data["t"].append(t_arr)
                data["qp"].append(qp_arr)
                data["qv"].append(qv_arr)
                data["qa"].append(qa_arr)
                data["tau"].append(tau_arr)
                data["m"].append(m_arr)
                data["c"].append(c_arr)
                data["g"].append(g_arr)
                data["p"].append(p_arr)
                data["pdot"].append(pdot_arr)

            except Exception as e:
                print(f"{csv_file} : {e}")

    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_filename, 'wb') as f:
        pickle.dump(data, f)

    print("\n==========================================")
    print(f"save to: {output_filename}")
    print(f" {data['labels'].count('train_labels')}  | : {data['labels'].count('test_labels')} ")
    print("==========================================")


if __name__ == "__main__":
    TRAIN_FOLDER = "train_labels"
    TEST_FOLDER = "test_labels"
    OUTPUT_PICKLE = "data/character_data.pickle"

    pack_csv_to_pickle(TRAIN_FOLDER, TEST_FOLDER, OUTPUT_PICKLE)