import pandas as pd
import pickle # pickleを直接使うように変更
import time
import numpy as np
from datetime import datetime
import os

# --- 設定値 ---
# GitHub Actionsのルートディレクトリからの相対パスで指定
# day5/演習1/ で保存されたモデルとデータへのパス
BASE_DIR_EX1 = "day5/演習1" # 演習1のベースディレクトリ
MODEL_PATH = os.path.join(BASE_DIR_EX1, "models", "titanic_model.pkl")
DATA_PATH = os.path.join(BASE_DIR_EX1, "data", "Titanic.csv") # 元データから推論用データを作成

N_ITERATIONS = 100
REPORTS_DIR = "inference_reports" # 結果保存ディレクトリ (演習3のコンテキスト)

def get_output_filename():
    """日付を含む出力ファイル名を生成する関数"""
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(REPORTS_DIR, f"inference_speed_result_{current_date}.txt")

def load_model(model_path):
    """pickleファイルからモデルをロードする関数"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Ensure 'day5/演習1/main.py' has been run to generate the model.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def prepare_inference_data(data_path, num_samples=50):
    """推論用のデータをロードし、前処理する関数"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Ensure 'Titanic.csv' is in '{os.path.join(BASE_DIR_EX1, 'data')}'")
    
    data = pd.read_csv(data_path)
    # day5/演習1/main.py の prepare_data と同様の前処理を実施
    data = data[["Pclass", "Sex", "Age", "Fare"]].copy() # Survived は推論時には不要
    
    # 欠損値処理 (演習1のdropna()に合わせるか、推論時の戦略を決める)
    # ここではAgeとFareの欠損値を平均値で埋める例 (演習1ではdropnaだったため、テストデータがない可能性)
    # 実際には学習時と同じ戦略で前処理する必要がある
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # SexのLabelEncoding (演習1ではfit_transformしていたが、推論時は学習時のエンコーダを使うか、固定マッピング)
    # ここでは簡易的にマッピング
    data["Sex"] = data["Sex"].map({'male': 0, 'female': 1}).fillna(0) # 未知の値は0(male)扱いなど

    # 型変換
    data["Pclass"] = data["Pclass"].astype(float)
    data["Sex"] = data["Sex"].astype(float)
    data["Age"] = data["Age"].astype(float)
    data["Fare"] = data["Fare"].astype(float)

    # サンプリング
    if len(data) == 0:
        raise ValueError("No data available after preprocessing. Check data and preprocessing steps.")
    
    sample_df = data.sample(min(len(data), num_samples), random_state=42)
    return sample_df


def measure_inference_time(model, data, iterations):
    """推論時間を測定する関数"""
    timings = []
    
    # ウォームアップ実行
    warmup_iterations = min(iterations // 10, 5)
    if warmup_iterations > 0 and not data.empty:
        for _ in range(warmup_iterations):
            _ = model.predict(data)

    # 実際の測定
    if not data.empty:
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model.predict(data)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
    else:
        print("Warning: No data to run inference on after preprocessing and sampling.")
        
    return timings

if __name__ == "__main__":
    output_filename = get_output_filename()

    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        print(f"Preparing inference data from {DATA_PATH}...")
        inference_data = prepare_inference_data(DATA_PATH)
        
        if not inference_data.empty:
            print(f"Measuring inference speed for {N_ITERATIONS} iterations using {len(inference_data)} samples per iteration...")
            durations = measure_inference_time(model, inference_data, N_ITERATIONS)
            
            if durations:
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
                result_summary = "Inference speed measurement could not be performed (no durations recorded)."
        else:
            result_summary = "Inference speed measurement could not be performed (no inference data)."

    except FileNotFoundError as e:
        print(f"Error: {e}")
        result_summary = f"Error during script execution: {e}\nMake sure the model and data files exist at the specified paths."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        result_summary = f"An unexpected error occurred: {e}"

    print("\n" + result_summary)
    
    with open(output_filename, "w") as f:
        f.write(result_summary)
    print(f"Results saved to {output_filename}")