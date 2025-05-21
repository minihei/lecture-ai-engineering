import pandas as pd
import joblib # モデルのロードに使っている場合はそのまま
import pickle # もしモデル保存がpickleならこちら
import time
import numpy as np
from datetime import datetime
import os
# from sklearn.preprocessing import LabelEncoder # 学習済みエンコーダを使うか、マッピングを固定

# 設定値
MODEL_PATH = "day5/演習1/models/titanic_model.pkl" # joblibで保存されている前提
DATA_PATH = "day5/演習1/data/Titanic.csv"
N_ITERATIONS = 100
REPORTS_DIR = "inference_reports" # 結果保存ディレクトリ

def get_output_filename():
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(REPORTS_DIR, f"inference_speed_result_{current_date}.txt")

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Ensure 'day5/演習1/main.py' has run and saved the model.")
    model = joblib.load(model_path) # joblibでロード
    return model

def prepare_inference_data(data_path, num_samples=50): # 関数名を変更して役割を明確に
    """推論用のデータをロードし、学習時と同様に前処理し、特徴量を選択する関数"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")
    
    raw_df = pd.read_csv(data_path)
    df_processed = raw_df.copy()

    # --- 学習スクリプト (day5/演習1/main.pyのprepare_data) と一貫した前処理 ---
    
    # 1. 'Sex' 列のエンコーディング (学習時と完全に同じ方法で)
    #    学習時に LabelEncoder().fit_transform() した場合、
    #    そのエンコーダを保存・ロードするか、マッピングを固定する。
    #    ここでは固定マッピングの例：
    if 'Sex' in df_processed.columns:
        df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})
        # マッピングできなかった場合や元々欠損していた場合の処理
        # (学習時のdropnaでSexの欠損は処理されていた可能性が高いが念のため)
        if df_processed['Sex'].isnull().any():
            # 例えば、学習データでの 'Sex' の最頻値で埋めるなど。
            # 学習時にdropnaしているので、推論時も Sex が male/female でない行は扱えないか、
            # あるいは学習時から Sex の欠損値/未知値の扱いを決めておく必要がある。
            # ここでは、もし欠損があれば特定の値 (例:0) で埋める。
            df_processed['Sex'].fillna(0, inplace=True)
    else:
        raise ValueError("Column 'Sex' not found in input data for inference.")

    # 2. 'Age', 'Fare' の欠損値処理
    #    学習スクリプトでは data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna() だった。
    #    推論時も同じように、これらのキーとなる特徴量に欠損がある行は使えない。
    #    ただし、'Survived' は推論時にはないので、'Pclass', 'Sex', 'Age', 'Fare' で判断。
    #    ここでは、学習時にdropna()していたので、それに倣うか、
    #    あるいは学習時から補完戦略をとるべき。
    #    今回は、学習時のdropnaを厳密に再現するのは難しいので、
    #    よく使われる中央値/平均値補完とし、学習時もそうだったと仮定するか、
    #    学習スクリプトも補完に合わせるのが理想。
    #    ここでは中央値補完を試みる。
    if 'Age' in df_processed.columns:
        df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    if 'Fare' in df_processed.columns:
        df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)
        
    # 3. 学習時に使用した特徴量のみを選択 (今回のエラーの直接的な解決策)
    features_for_model = ["Pclass", "Sex", "Age", "Fare"]
    
    # 選択前に必要な列が存在するか最終確認
    missing_cols = [col for col in features_for_model if col not in df_processed.columns]
    if missing_cols:
        # このエラーは、Sex, Age, Fare の前処理がうまくいっていない場合に発生する可能性
        raise ValueError(f"After preprocessing, required feature columns are missing: {missing_cols}. Available: {df_processed.columns.tolist()}")
        
    df_final_features = df_processed[features_for_model].copy()

    # 4. 学習時と同様の型変換
    try:
        df_final_features["Pclass"] = df_final_features["Pclass"].astype(float)
        df_final_features["Sex"] = df_final_features["Sex"].astype(float)
        df_final_features["Age"] = df_final_features["Age"].astype(float)
        df_final_features["Fare"] = df_final_features["Fare"].astype(float)
    except Exception as e:
        print(f"Error during type conversion: {e}")
        print("Data types before conversion attempt:")
        print(df_final_features.dtypes)
        raise

    # 5. サンプル数を調整 (推論スピード測定用)
    if len(df_final_features) == 0:
        raise ValueError("No data available for inference after all preprocessing and feature selection.")
        
    # サンプリングする前に、列が正しいか最終確認
    print(f"Columns in df_final_features before sampling: {df_final_features.columns.tolist()}")
    print(f"Data types in df_final_features before sampling:\n{df_final_features.dtypes}")


    sample_df = df_final_features.sample(min(len(df_final_features), num_samples), random_state=42)
    
    print(f"Shape of sampled data for inference: {sample_df.shape}")
    print(f"Columns in sampled data for inference: {sample_df.columns.tolist()}")
    
    return sample_df


def measure_inference_time(model, data, iterations):
    timings = []
    if data.empty:
        print("Warning: No data provided to measure_inference_time. Skipping measurement.")
        return timings # 空のリストを返す

    # ウォームアップ実行
    warmup_count = min(iterations // 10, 5)
    if warmup_count > 0:
        print(f"Performing {warmup_count} warmup iterations...")
        for _ in range(warmup_count):
            try:
                _ = model.predict(data)
            except Exception as e:
                print(f"Error during warmup prediction: {e}")
                print(f"Data columns during warmup: {data.columns.tolist()}")
                print(f"Data dtypes during warmup:\n{data.dtypes}")
                raise # エラーを再スローして処理を止める
    
    # 実際の測定
    print(f"Performing {iterations} measurement iterations...")
    for i in range(iterations):
        start_time = time.perf_counter()
        try:
            _ = model.predict(data)
        except Exception as e:
            print(f"Error during prediction at iteration {i+1}: {e}")
            print(f"Data columns at iteration {i+1}: {data.columns.tolist()}")
            print(f"Data dtypes at iteration {i+1}:\n{data.dtypes}")
            raise # エラーを再スロー
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    return timings

if __name__ == "__main__":
    output_filename = get_output_filename()
    result_summary = "" # 初期化

    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        print(f"Preparing inference data from {DATA_PATH}...")
        # load_data 関数を prepare_inference_data に変更
        inference_data = prepare_inference_data(DATA_PATH, num_samples=N_ITERATIONS) # サンプル数をN_ITERATIONSと同じにするか検討
        
        if not inference_data.empty:
            print(f"Measuring inference speed for {N_ITERATIONS} iterations...")
            durations = measure_inference_time(model, inference_data, N_ITERATIONS)
            
            if durations: # durationsが空でないことを確認
                avg_duration_ms = np.mean(durations) * 1000
                median_duration_ms = np.median(durations) * 1000
                p95_duration_ms = np.percentile(durations, 95) * 1000
                total_duration_s = sum(durations)
                
                result_summary = (
                    f"Inference Speed Measurement (executed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):\n"
                    f"Model used: {MODEL_PATH}\n"
                    f"Data samples used for each predict call: {len(inference_data)}\n"
                    f"Number of iterations: {N_ITERATIONS}\n"
                    f"Average inference time: {avg_duration_ms:.4f} ms\n"
                    f"Median inference time: {median_duration_ms:.4f} ms\n"
                    f"95th percentile inference time: {p95_duration_ms:.4f} ms\n"
                    f"Total time for {N_ITERATIONS} iterations: {total_duration_s:.4f} s\n"
                )
            else:
                result_summary = "Inference speed measurement completed, but no durations were recorded (possibly due to empty data or errors during predict)."
        else:
            result_summary = "No data available for inference after preprocessing. Speed measurement not performed."

    except FileNotFoundError as e:
        print(f"Error: {e}")
        result_summary = f"Script Error: {e}. Ensure model and data files exist."
    except ValueError as e:
        print(f"ValueError during script execution: {e}") # このエラーが再度出るか確認
        result_summary = f"Script Error: {e}."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        result_summary = f"An unexpected script error occurred: {e}."

    print("\n" + result_summary)
    
    with open(output_filename, "w") as f:
        f.write(result_summary)
    print(f"Results saved to {output_filename}")