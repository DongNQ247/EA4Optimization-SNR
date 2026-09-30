"""
Test Convergence for GA, DE, PSO Algorithms
--------------------------------------------
Chạy kiểm thử 100 lần (hoặc 50) cho mỗi thuật toán với các kịch bản pop_size khác nhau.
Lọc các fitness_history hợp lệ (hội tụ) và tính trung bình để vẽ biểu đồ.

Author: SNR Optimization Team
Date: January 21, 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from algorithm import genetic_algorithm, de, pso
from problem import Problem
from config import R_R, X_R, R_L, X_L

# ============================================================================
# CONFIGURATION
# ============================================================================

# Tham số kiểm thử
POP_SIZES = [30, 50, 100]  # Các kịch bản pop_size cần test
MAX_ITER = 20              # Số vòng lặp cho tất cả thuật toán
NUM_RUNS = 100             # Số lần chạy cho mỗi kịch bản (có thể giảm xuống 50)
EPSILON = 0.2              # Ngưỡng hội tụ (dB)
CONVERGENCE_START = 14     # Bắt đầu kiểm tra hội tụ từ phần tử thứ 15 (index 14)
CONVERGENCE_END = 20       # Đến phần tử thứ 20 (index 19)

OUTPUT_DIR = "result"

# Tham số cho các thuật toán
GA_PARAMS = {"pc": 0.5, "pm": 0.5}
DE_PARAMS = {"F": 0.8, "CR": 0.9}
PSO_PARAMS = {"w": 0.7, "c1": 1.5, "c2": 1.5}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_convergence(fitness_history, snr_theorem, epsilon=EPSILON):
    """
    Kiểm tra xem fitness_history có hội tụ hay không.
    
    Tiêu chí: Các giá trị từ phần tử 15-20 (index 14-19) phải gần với SNR_theorem
    trong khoảng epsilon.
    
    Parameters:
    -----------
    fitness_history : list
        Lịch sử fitness qua các vòng lặp
    snr_theorem : float
        Giá trị SNR lý thuyết tối đa
    epsilon : float
        Ngưỡng chấp nhận (dB)
    
    Returns:
    --------
    bool : True nếu hội tụ, False nếu không
    """
    if len(fitness_history) < CONVERGENCE_END:
        return False
    
    # Lấy các giá trị từ phần tử 15-20
    convergence_values = fitness_history[CONVERGENCE_START:CONVERGENCE_END]
    
    # Kiểm tra xem tất cả các giá trị có trong khoảng epsilon so với SNR_theorem
    for value in convergence_values:
        if abs(value - snr_theorem) > epsilon:
            return False
    
    return True


def run_algorithm_multiple_times(algorithm_name, problem, pop_size, num_runs, snr_theorem):
    """
    Chạy một thuật toán nhiều lần và lọc kết quả hợp lệ.
    
    Parameters:
    -----------
    algorithm_name : str
        Tên thuật toán ('GA', 'DE', 'PSO')
    problem : Problem
        Đối tượng bài toán
    pop_size : int
        Kích thước quần thể
    num_runs : int
        Số lần chạy
    snr_theorem : float
        Giá trị SNR lý thuyết
    
    Returns:
    --------
    list : Danh sách các fitness_history hợp lệ
    """
    valid_fitness_histories = []
    
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm_name}, Pop Size: {pop_size}")
    print(f"{'='*60}")
    
    for run_id in range(num_runs):
        # Chọn thuật toán
        if algorithm_name == 'GA':
            _, fitness_history = genetic_algorithm(
                problem=problem,
                pop_size=pop_size,
                max_iter=MAX_ITER,
                pc=GA_PARAMS['pc'],
                pm=GA_PARAMS['pm'],
                patience=MAX_ITER,  # Không dừng sớm
                history=False
            )
        elif algorithm_name == 'DE':
            _, fitness_history = de(
                problem=problem,
                pop_size=pop_size,
                max_iter=MAX_ITER,
                F=DE_PARAMS['F'],
                CR=DE_PARAMS['CR'],
                patience=MAX_ITER,  # Không dừng sớm
                history=False
            )
        elif algorithm_name == 'PSO':
            _, fitness_history = pso(
                problem=problem,
                pop_size=pop_size,
                max_iter=MAX_ITER,
                w=PSO_PARAMS['w'],
                c1=PSO_PARAMS['c1'],
                c2=PSO_PARAMS['c2'],
                patience=MAX_ITER,  # Không dừng sớm
                history=False
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Kiểm tra hội tụ
        is_converged = check_convergence(fitness_history, snr_theorem, EPSILON)
        
        if is_converged:
            valid_fitness_histories.append(fitness_history)
        
        # In tiến trình
        if (run_id + 1) % 10 == 0:
            valid_count = len(valid_fitness_histories)
            print(f"  Progress: {run_id + 1}/{num_runs} runs | Valid: {valid_count}")
    
    valid_count = len(valid_fitness_histories)
    print(f"\n  ✓ Completed: {valid_count}/{num_runs} valid runs")
    
    return valid_fitness_histories


def calculate_average_fitness_history(valid_fitness_histories):
    """
    Tính trung bình fitness_history theo từng vòng lặp.
    
    Parameters:
    -----------
    valid_fitness_histories : list of lists
        Danh sách các fitness_history hợp lệ
    
    Returns:
    --------
    list : avg_fitness_history
    """
    if len(valid_fitness_histories) == 0:
        return []
    
    # Chuyển đổi thành numpy array để tính toán dễ dàng
    histories_array = np.array(valid_fitness_histories)
    
    # Tính trung bình theo cột (axis=0)
    avg_fitness_history = np.mean(histories_array, axis=0)
    
    return avg_fitness_history.tolist()


def save_matlab_data(results_dict, snr_theorem, output_dir=OUTPUT_DIR):
    """
    Lưu dữ liệu dưới dạng .mat để MATLAB có thể đọc và vẽ lại biểu đồ.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary chứa kết quả: {(algorithm, pop_size): avg_fitness_history}
    snr_theorem : float
        Giá trị SNR lý thuyết
    output_dir : str
        Thư mục đầu ra
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu cho từng thuật toán
    for algorithm in ['GA', 'DE', 'PSO']:
        for pop_size in POP_SIZES:
            key = (algorithm, pop_size)
            if key in results_dict and len(results_dict[key]) > 0:
                avg_history = results_dict[key]
                
                # Tạo dictionary dữ liệu cho MATLAB
                matlab_data = {
                    'iterations': np.arange(len(avg_history)),
                    'snr_values': np.array(avg_history),
                    'snr_theorem': snr_theorem,
                    'algorithm': algorithm,
                    'pop_size': pop_size,
                    'max_iter': MAX_ITER
                }
                
                # Lưu file .mat
                filename = f"convergence_{algorithm}_N{pop_size}.mat"
                save_path = os.path.join(output_dir, filename)
                scipy.io.savemat(save_path, matlab_data)
                print(f"  ✓ Saved MATLAB data: {save_path}")
    
    # Lưu file tổng hợp cho comparison plot (chỉ N=50)
    comparison_data = {
        'snr_theorem': snr_theorem,
        'max_iter': MAX_ITER,
        'iterations': np.arange(MAX_ITER)
    }
    
    for algorithm in ['GA', 'DE', 'PSO']:
        key = (algorithm, 50)  # Chỉ lấy N=50
        if key in results_dict and len(results_dict[key]) > 0:
            comparison_data[f'{algorithm}_snr'] = np.array(results_dict[key])
    
    save_path = os.path.join(output_dir, "convergence_comparison_all.mat")
    scipy.io.savemat(save_path, comparison_data)
    print(f"  ✓ Saved MATLAB comparison data: {save_path}")


def plot_comparison_all_algorithms(results_dict, snr_theorem):
    """
    Vẽ biểu đồ so sánh tất cả các thuật toán và pop_size.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary chứa kết quả: {(algorithm, pop_size): avg_fitness_history}
    snr_theorem : float
        Giá trị SNR lý thuyết
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Màu sắc cho các thuật toán
    colors = {
        'GA': '#1f77b4',   # Xanh dương
        'DE': '#d62728',   # Đỏ
        'PSO': '#2ca02c'   # Xanh lá
    }
    
    # Kiểu đường cho các pop_size
    linestyles = {
        30: '-',      # Nét liền
        50: '--',     # Nét đứt
        100: ':'      # Nét chấm
    }
    
    # Vẽ biểu đồ tổng hợp
    plt.figure(figsize=(12, 7))
    
    # Vẽ đường lý thuyết (có label trong legend)
    plt.axhline(y=snr_theorem, color='black', linestyle='-', 
                linewidth=2.5, label='SNR_theorem', zorder=5)
    
    # CHỈ vẽ các trường hợp N=50
    target_pop_size = 50
    for algorithm in ['GA', 'DE', 'PSO']:
        key = (algorithm, target_pop_size)
        if key in results_dict and len(results_dict[key]) > 0:
            avg_history = results_dict[key]
            label = f"{algorithm} (N={target_pop_size})"
            plt.plot(
                range(len(avg_history)),
                avg_history,
                label=label,
                color=colors[algorithm],
                linestyle='-',
                linewidth=2,
                alpha=0.8
            )
    
    # Thêm annotation cho SNR_theorem bên phải đường ngang
    plt.text(MAX_ITER, snr_theorem, f'  {snr_theorem:.2f} dB', 
             verticalalignment='center', fontsize=11, 
             color='black', fontweight='bold')
    
    plt.xlabel("Iteration", fontsize=14, fontweight='bold')
    plt.ylabel("Average SNR (dB)", fontsize=14, fontweight='bold')
    plt.title("Algorithm Convergence Comparison", 
              fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # Cài đặt trục hoành: bắt đầu từ 0, chia đơn vị 2
    plt.xlim(0, MAX_ITER)
    plt.xticks(range(0, MAX_ITER + 1, 2))
    plt.tight_layout()
    
    # Lưu PNG
    save_path_png = os.path.join(OUTPUT_DIR, "convergence_comparison_all.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Saved comparison plot PNG: {save_path_png}")
    
    # Lưu SVG
    save_path_svg = os.path.join(OUTPUT_DIR, "convergence_comparison_all.svg")
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    print(f"✓ Saved comparison plot SVG: {save_path_svg}")
    
    # Lưu FIG
    save_path_fig = os.path.join(OUTPUT_DIR, "convergence_comparison_all.fig")
    import pickle
    fig = plt.gcf()
    with open(save_path_fig, 'wb') as f:
        pickle.dump(fig, f)
    print(f"✓ Saved comparison plot FIG: {save_path_fig}")
    print(f"{'='*60}")
    
    plt.close()


def plot_individual_algorithm(algorithm_name, results_dict, snr_theorem):
    """
    Vẽ biểu đồ riêng cho một thuật toán với các pop_size khác nhau.
    
    Parameters:
    -----------
    algorithm_name : str
        Tên thuật toán
    results_dict : dict
        Dictionary chứa kết quả
    snr_theorem : float
        Giá trị SNR lý thuyết
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Vẽ đường lý thuyết (có label trong legend)
    plt.axhline(y=snr_theorem, color='black', linestyle='-', 
                linewidth=2.5, label='SNR_theorem', zorder=5)
    
    # Màu sắc cho các pop_size
    colors_pop = {30: '#1f77b4', 50: '#ff7f0e', 100: '#2ca02c'}
    
    # Vẽ từng đường pop_size
    for pop_size in POP_SIZES:
        key = (algorithm_name, pop_size)
        if key in results_dict and len(results_dict[key]) > 0:
            avg_history = results_dict[key]
            plt.plot(
                range(len(avg_history)),
                avg_history,
                label=f"N={pop_size}",
                color=colors_pop[pop_size],
                linewidth=2.5,
                marker='o',
                markersize=4,
                alpha=0.9
            )
    
    # Thêm annotation cho SNR_theorem bên phải đường ngang
    plt.text(MAX_ITER, snr_theorem, f'  {snr_theorem:.2f} dB', 
             verticalalignment='center', fontsize=11, 
             color='black', fontweight='bold')
    
    plt.xlabel("Iteration", fontsize=14, fontweight='bold')
    plt.ylabel("Average SNR (dB)", fontsize=14, fontweight='bold')
    plt.title(f"{algorithm_name} Convergence Analysis", 
              fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Cài đặt trục hoành: bắt đầu từ 0, chia đơn vị 2
    plt.xlim(0, MAX_ITER)
    plt.xticks(range(0, MAX_ITER + 1, 2))
    plt.tight_layout()
    
    # Lưu PNG
    save_path_png = os.path.join(OUTPUT_DIR, f"convergence_{algorithm_name}.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {algorithm_name} plot PNG: {save_path_png}")
    
    # Lưu SVG
    save_path_svg = os.path.join(OUTPUT_DIR, f"convergence_{algorithm_name}.svg")
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    print(f"  ✓ Saved {algorithm_name} plot SVG: {save_path_svg}")
    
    # Lưu FIG
    save_path_fig = os.path.join(OUTPUT_DIR, f"convergence_{algorithm_name}.fig")
    import pickle
    fig = plt.gcf()
    with open(save_path_fig, 'wb') as f:
        pickle.dump(fig, f)
    print(f"  ✓ Saved {algorithm_name} plot FIG: {save_path_fig}")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Hàm chính để chạy kiểm thử."""
    
    print("\n" + "="*60)
    print("CONVERGENCE TEST - GA, DE, PSO Algorithms")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Pop sizes: {POP_SIZES}")
    print(f"  - Max iterations: {MAX_ITER}")
    print(f"  - Number of runs: {NUM_RUNS}")
    print(f"  - Convergence epsilon: {EPSILON} dB")
    print(f"  - Convergence check: elements {CONVERGENCE_START+1}-{CONVERGENCE_END}")
    print("="*60)
    
    # Khởi tạo bài toán
    problem = Problem(R_R, X_R, R_L, X_L)
    snr_theorem = problem.SNR_theorem(show=False)
    
    print(f"\nSNR Theorem (Theoretical Max): {snr_theorem:.4f} dB")
    
    # Dictionary lưu kết quả
    results = {}
    
    # Danh sách thuật toán
    algorithms = ['GA', 'DE', 'PSO']
    
    # Chạy kiểm thử cho tất cả các thuật toán và pop_size
    for algorithm in algorithms:
        for pop_size in POP_SIZES:
            # Chạy nhiều lần
            valid_histories = run_algorithm_multiple_times(
                algorithm_name=algorithm,
                problem=problem,
                pop_size=pop_size,
                num_runs=NUM_RUNS,
                snr_theorem=snr_theorem
            )
            
            # Tính trung bình
            if len(valid_histories) > 0:
                avg_history = calculate_average_fitness_history(valid_histories)
                results[(algorithm, pop_size)] = avg_history
                
                print(f"  → Average fitness history calculated ({len(avg_history)} iterations)")
            else:
                results[(algorithm, pop_size)] = []
                print(f"  ⚠ Warning: No valid runs found!")
    
    # Vẽ biểu đồ tổng hợp
    print("\n" + "="*60)
    print("GENERATING PLOTS...")
    print("="*60)
    
    plot_comparison_all_algorithms(results, snr_theorem)
    
    # Vẽ biểu đồ riêng cho từng thuật toán
    for algorithm in algorithms:
        plot_individual_algorithm(algorithm, results, snr_theorem)
    
    # Lưu dữ liệu MATLAB
    print("\n" + "="*60)
    print("EXPORTING MATLAB DATA...")
    print("="*60)
    
    save_matlab_data(results, snr_theorem)
    
    # In thống kê cuối cùng
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for algorithm in algorithms:
        print(f"\n{algorithm}:")
        for pop_size in POP_SIZES:
            key = (algorithm, pop_size)
            if key in results and len(results[key]) > 0:
                avg_history = results[key]
                final_snr = avg_history[-1]
                diff = abs(final_snr - snr_theorem)
                print(f"  Pop Size {pop_size:3d}: Final SNR = {final_snr:.4f} dB (Δ = {diff:.4f} dB)")
            else:
                print(f"  Pop Size {pop_size:3d}: No valid data")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS COMPLETED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
