import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_steering_error(file_path, threshold=0.15):
    try:
        df = pd.read_csv(file_path)
        
        # [수정됨] 컬럼 매핑
        col_target = 'angular_z'        # 정답 (Ground Truth)
        col_pred = 'predicted_steering' # 모델 예측값
        
        # 오차 계산
        df['error'] = df[col_pred] - df[col_target]
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 구간 분류 (직선 vs 코너)
    # 각속도(angular_z)의 절댓값이 작으면 직진, 크면 회전으로 간주
    mask_straight = df[col_target].abs() < threshold
    mask_corner = df[col_target].abs() >= threshold

    straight_errors = df.loc[mask_straight, 'error']
    corner_errors = df.loc[mask_corner, 'error']

    # 시각화
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")

    # (1) 직선 구간 (Straight)
    plt.subplot(1, 2, 1)
    sns.histplot(straight_errors, bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title(f'Straight Error (|ang_z| < {threshold})', fontsize=14, fontweight='bold')
    plt.xlabel('Error (Predicted - Truth)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    mean_s = straight_errors.mean()
    std_s = straight_errors.std()
    plt.text(0.95, 0.95, f'Mean: {mean_s:.4f}\nStd: {std_s:.4f}', 
             transform=plt.gca().transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # (2) 코너 구간 (Corner)
    plt.subplot(1, 2, 2)
    sns.histplot(corner_errors, bins=30, kde=True, color='salmon', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.title(f'Corner Error (|ang_z| >= {threshold})', fontsize=14, fontweight='bold')
    plt.xlabel('Error (Predicted - Truth)', fontsize=12)
    plt.ylabel('Count', fontsize=12)

    mean_c = corner_errors.mean()
    std_c = corner_errors.std()
    plt.text(0.95, 0.95, f'Mean: {mean_c:.4f}\nStd: {std_c:.4f}', 
             transform=plt.gca().transAxes, ha='right', va='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_name = 'steering_analysis.png'
    plt.savefig(save_name, dpi=300) 
    print(f"💾 그래프가 저장되었습니다: {os.path.abspath(save_name)}")
    
    print("="*40)
    print(f"📊 분석 결과")
    print(f" - 직진 구간 평균 오차(Bias): {mean_s:.4f}")
    print(f" - 코너 구간 표준 편차(Std):  {std_c:.4f}")
    print("="*40)

# 실행
analyze_steering_error('driving_log_with_pred2.csv', threshold=0.2)