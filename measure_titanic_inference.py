import pandas as pd
import joblib
import time
import numpy as np
from datetime import datetime

# 設定値
MODEL_PATH = "day5/演習1/models/titanic_model.pkl"
DATA_PATH = "day5/演習1/data/Titanic.csv"
N_ITERATIONS = 100  # 測定の繰り返し回数

def get_output_filename():
    """日付を含む出力ファイル名を生成する関数"""
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S") # YYYYMMDD_HHMMSS形式
    return f"inference_speed_result_{current_date}.txt"

def load_model(model_path):
    """モデルをロードする関数"""
    # 例: scikit-learnのモデルの場合
    model = joblib.load(model_path)
    return model

def load_data(data_path):
    """推論用のデータをロードし、前処理する関数 (一部をサンプリングするなど)"""
    df = pd.read_csv(data_path)
    # 必要であれば、推論に必要な特徴量選択や前処理を行う
    # ここでは例として、最初の5行を推論対象とします (実際のテストではより多くのデータを使うことを推奨)
    # 全データでテストすると時間がかかりすぎる場合、代表的なサンプルでテストすることも考慮
    sample_df = df.sample(min(len(df), 50), random_state=42) # 例: 最大50件でテスト
    # 特徴量Xとターゲットyに分ける (推論ではXのみ使用)
    # X = sample_df.drop('Survived', axis=1, errors='ignore') # 'Survived'列がなければ無視
    X = sample_df # ここでは簡単のため、前処理済みのデータフレーム全体をXと仮定
    return X


def measure_inference_time(model, data, iterations):
    """推論時間を測定する関数"""
    timings = []
    
    # ウォームアップ実行 (最初の数回はキャッシュなどの影響で遅いことがあるため)
    for _ in range(min(iterations // 10, 5)): # 例: 10%か最大5回
        _ = model.predict(data)

    # 実際の測定
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = model.predict(data) # ここが実際の推論処理
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    
    return timings

if __name__ == "__main__":
    output_filename = get_output_filename() # ここで 'output_filename' を定義
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    print(f"Loading data from {DATA_PATH}...")
    inference_data = load_data(DATA_PATH)
    
    print(f"Measuring inference speed for {N_ITERATIONS} iterations...")
    durations = measure_inference_time(model, inference_data, N_ITERATIONS)
    
    avg_duration_ms = np.mean(durations) * 1000
    median_duration_ms = np.median(durations) * 1000
    p95_duration_ms = np.percentile(durations, 95) * 1000
    
    result_summary = (
        f"Inference Speed Measurement:\n"
        f"Data samples used for each predict call: {len(inference_data)}\n"
        f"Number of iterations: {N_ITERATIONS}\n"
        f"Average inference time: {avg_duration_ms:.4f} ms\n"
        f"Median inference time: {median_duration_ms:.4f} ms\n"
        f"95th percentile inference time: {p95_duration_ms:.4f} ms\n"
        f"Total time for {N_ITERATIONS} iterations: {sum(durations):.4f} s\n"
    )
    
    print("\n" + result_summary)
    
    # 結果をファイルに保存
    with open(output_filename, "w") as f:   # ここで 'output_filename' を使用
        f.write(result_summary)
    print(f"Results saved to {output_filename}")