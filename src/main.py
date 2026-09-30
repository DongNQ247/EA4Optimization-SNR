"""
SNR Optimization using Evolutionary Algorithms
------------------------------------------------
This module implements and compares multiple optimization algorithms
(GA, DE, PSO) for Signal-to-Noise Ratio (SNR) maximization in impedance matching.

Author: SNR Optimization Team
Date: January 19, 2026
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from algorithm import genetic_algorithm, de, pso
from problem   import Problem
from config    import g, Z_RT, S_IT, N_na, T, k, R_R, X_R, R_L, X_L

# ============================================================================
# CONFIGURATION
# ============================================================================

# Cấu hình chung cho biểu đồ xuất bản
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,      
    'axes.titlesize': 14,
    'xtick.labelsize': 12,     
    'ytick.labelsize': 12,     
    'legend.fontsize': 11,
    'lines.linewidth': 2.0,    
    'figure.autolayout': True  
})

# Cấu hình thuật toán
ALGORITHMS = ['GA', 'DE', 'PSO']  # Các thuật toán sẽ chạy
OUTPUT_DIR = "result"
MAX_ITER = 1000
PATIENCE = 200

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_publication_plot(history, filename, output_dir=OUTPUT_DIR, 
                          is_comparison=False, all_histories=None, 
                          theoretical_max=None, labels=None, colors=None, title=None):
    """
    Lưu biểu đồ với định dạng xuất bản khoa học
    
    Parameters:
    -----------
    history : list
        Lịch sử fitness của một thuật toán
    filename : str
        Tên file lưu
    output_dir : str
        Thư mục đầu ra
    is_comparison : bool
        True nếu so sánh nhiều thuật toán
    all_histories : dict
        Dictionary chứa lịch sử của nhiều thuật toán
    theoretical_max : float
        Giá trị SNR lý thuyết tối đa
    labels : dict
        Nhãn cho từng đường
    colors : dict
        Màu sắc cho từng đường
    title : str
        Tiêu đề cho biểu đồ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if is_comparison and all_histories:
        # Vẽ đường lý thuyết - nét liền màu đen
        if theoretical_max is not None:
            ax.axhline(y=theoretical_max, color='black', linestyle='-', 
                       linewidth=2, label='Theoretical Max', zorder=5)
        
        # Màu sắc mặc định cho các thuật toán
        default_colors = {
            'GA': '#1f77b4',  # Xanh dương
            'DE': '#d62728',  # Đỏ
            'PSO': '#2ca02c'  # Xanh lá
        }
        
        # Vẽ từng thuật toán - nét đứt
        for alg_name, hist in all_histories.items():
            color = colors.get(alg_name, default_colors.get(alg_name, '#888888')) if colors else default_colors.get(alg_name, '#888888')
            label = labels.get(alg_name, alg_name) if labels else alg_name
            ax.plot(hist, label=label, color=color, linewidth=2, linestyle='--', alpha=0.9)
            
        ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor='gray', shadow=True)
        
    else:
        # Vẽ một đường duy nhất
        ax.plot(history, color='#1f77b4', linewidth=2)         
    
    ax.set_xlabel("Iteration", fontweight='bold')
    ax.set_ylabel("SNR (dB)", fontweight='bold')
    
    # Thêm title nếu có
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.6)
    # Giới hạn trục iteration từ 0 đến 200
    ax.set_xlim(0, 20)
    
    # Lưu ảnh PNG
    save_path_png = os.path.join(output_dir, filename)
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG: {save_path_png}")
    
    # Lưu ảnh SVG (vector format - mở được bằng Illustrator, Inkscape)
    svg_filename = filename.replace('.png', '.svg')
    save_path_svg = os.path.join(output_dir, svg_filename)
    plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
    print(f"  ✓ Saved SVG: {save_path_svg}")
    
    # Lưu ảnh FIG (pickle format - có thể mở lại và chỉnh sửa)
    fig_filename = filename.replace('.png', '.fig')
    save_path_fig = os.path.join(output_dir, fig_filename)
    import pickle
    with open(save_path_fig, 'wb') as f:
        pickle.dump(fig, f)
    print(f"  ✓ Saved FIG: {save_path_fig}")
    
    plt.close()


def print_results_table(results, theoretical_max):
    """
    In bảng kết quả định dạng khoa học
    
    Parameters:
    -----------
    results : list
        Danh sách các kết quả (algorithm_name, case_name, Z_R, Z_L, SNR)
    theoretical_max : float
        Giá trị SNR lý thuyết tối đa
    """
    print("\n" + "="*95)
    print(f"{'Algorithm':<12} | {'Case':<12} | {'Z_R':<25} | {'Z_L':<25} | {'SNR (dB)':<10}")
    print("-" * 95)
    for res in results:
        alg, case, zr, zl, snr = res
        zr_str = f"{zr.real:.2f}{zr.imag:+.2f}j"
        zl_str = f"{zl.real:.2f}{zl.imag:+.2f}j"
        print(f"{alg:<12} | {case:<12} | {zr_str:<25} | {zl_str:<25} | {snr:<10.4f}")
    print("="*95)
    print(f"Theoretical Maximum SNR: {theoretical_max:.4f} dB")
    print()


def run_algorithm(algorithm_name, problem, params, case_name):
    """
    Chạy một thuật toán tối ưu hóa
    
    Parameters:
    -----------
    algorithm_name : str
        Tên thuật toán ('GA', 'DE', 'PSO')
    problem : Problem
        Bài toán tối ưu hóa
    params : dict
        Tham số của thuật toán
    case_name : str
        Tên case thử nghiệm
        
    Returns:
    --------
    tuple : ((Z_R, Z_L), history)
        Kết quả tối ưu và lịch sử fitness
    """
    if algorithm_name == 'GA':
        return genetic_algorithm(
            problem,
            pop_size=params['pop_size'],
            max_iter=params['max_iter'],
            pc=params.get('pc', 0.95),
            pm=params.get('pm', 0.05),
            patience=params['patience'],
            history=False,
            case_name=case_name
        )
    elif algorithm_name == 'DE':
        return de(
            problem,
            pop_size=params['pop_size'],
            max_iter=params['max_iter'],
            F=params.get('F', 0.8),
            CR=params.get('CR', 0.9),
            patience=params['patience'],
            history=False,
            case_name=case_name
        )
    elif algorithm_name == 'PSO':
        return pso(
            problem,
            pop_size=params['pop_size'],
            max_iter=params['max_iter'],
            w=params.get('w', 0.7),
            c1=params.get('c1', 1.5),
            c2=params.get('c2', 1.5),
            patience=params['patience'],
            history=False,
            case_name=case_name
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Chương trình chính thực hiện thí nghiệm tối ưu hóa SNR"""
    
    print("="*95)
    print(" "*25 + "SNR OPTIMIZATION USING EVOLUTIONARY ALGORITHMS")
    print("="*95)
    
    # Khởi tạo bài toán
    problem = Problem(R_R, X_R, R_L, X_L)
    SNR_theorem = problem.SNR_theorem(show=False)
    
    print(f"\n📊 Theoretical Maximum SNR: {SNR_theorem:.4f} dB\n")
    
    # Định nghĩa các kịch bản thử nghiệm
    test_cases = [
        {
            "name": "Small Pop",
            "params": {
                "pop_size": 30,
                "max_iter": MAX_ITER,
                "pc": 0.8, "pm": 0.2,  # GA (khai thác mạnh, ít đột biến)
                # DE: giảm CR, giảm nhẹ F để bước đi đa dạng hơn và chậm hội tụ
                "F": 0.5, "CR": 0.3,
                # PSO: tăng w để giữ quán tính, giảm c1,c2 để bớt bị kéo mạnh về gbest/pbest
                "w": 1.0, "c1": 0.2, "c2": 0.2,
                "patience": PATIENCE
            }
        },
        {
            "name": "Medium Pop",
            "params": {
                "pop_size": 50,
                "max_iter": MAX_ITER,
                "pc": 0.8, "pm": 0.2,
                "F": 0.5, "CR": 0.3,
                "w": 1.0, "c1": 0.2, "c2": 0.2,
                "patience": PATIENCE
            }
        },
        {
            "name": "Large Pop",
            "params": {
                "pop_size": 100,
                "max_iter": MAX_ITER,
                "pc": 0.8, "pm": 0.2,
                "F": 0.5, "CR": 0.3,
                "w": 1.0, "c1": 0.2, "c2": 0.2,
                "patience": PATIENCE
            }
        }
    ]
    
    # Lưu trữ kết quả
    all_results = []
    algorithm_histories = {alg: {} for alg in ALGORITHMS}
    
    # Chạy các thử nghiệm
    print("🚀 Starting Experiments...\n")
    
    for case in test_cases:
        case_name = case["name"]
        params = case["params"]
        
        print(f"{'─'*95}")
        print(f"📌 Running Case: {case_name} (Population Size: {params['pop_size']})")
        print(f"{'─'*95}")
        
        for alg_name in ALGORITHMS:
            print(f"  ⚙️  Algorithm: {alg_name}...", end=" ")
            
            (Z_R_res, Z_L_res), history = run_algorithm(alg_name, problem, params, f"{alg_name}_{case_name}")
            
            SNR_result = problem.SNR(Z_R_res, Z_L_res)
            all_results.append((alg_name, case_name, Z_R_res, Z_L_res, SNR_result))
            algorithm_histories[alg_name][case_name] = history
            
            print(f"SNR = {SNR_result:.4f} dB")
            
            # Lưu biểu đồ riêng cho từng case
            filename = f"{alg_name.lower()}_{case_name.lower().replace(' ', '_')}.png"
            plot_title = f"{alg_name} - {case_name} (N={params['pop_size']})"
            save_publication_plot(history, filename, title=plot_title)
        
        print()
    
    # In bảng kết quả tổng hợp
    print("\n" + "🎯 FINAL RESULTS ".center(95, "="))
    print_results_table(all_results, SNR_theorem)
    
    # Vẽ biểu đồ so sánh theo từng thuật toán (nhóm các pop_size lại)
    print("📊 Generating Algorithm-wise Comparison Plots...\n")
    
    for alg_name in ALGORITHMS:
        print(f"  📈 Creating plot for {alg_name}...")
        
        # Lấy histories của thuật toán này với các pop_size khác nhau
        histories_for_alg = algorithm_histories[alg_name]
        
        # Tạo labels với thông tin pop_size
        labels = {case_name: f"{case_name} (N={case['params']['pop_size']})" 
                  for case in test_cases 
                  for case_name in [case['name']]}
        
        # Màu sắc khác nhau cho mỗi pop_size (N = 30, 50, 100)
        # N=30  -> đỏ, N=50 -> cam, N=100 -> xanh dương
        colors = {
            test_cases[0]["name"]: 'red',      # Small Pop  (N=30)
            test_cases[1]["name"]: 'orange',   # Medium Pop (N=50)
            test_cases[2]["name"]: 'blue'      # Large Pop  (N=100)
        }
        
        filename = f"{alg_name.lower()}_comparison_popsize.png"
        plot_title = f"{alg_name} Algorithm - Population Size Comparison"
        
        save_publication_plot(
            history=None,
            filename=filename,
            is_comparison=True,
            all_histories=histories_for_alg,
            theoretical_max=SNR_theorem,
            labels=labels,
            colors=colors,
            title=plot_title
        )
    
    # Vẽ biểu đồ so sánh tất cả các thuật toán (với pop_size tốt nhất)
    print("\n📊 Generating Overall Comparison Plot (All Algorithms)...\n")
    best_case_name = test_cases[1]["name"]  # Chọn Medium Pop làm đại diện
    overall_histories = {alg: algorithm_histories[alg][best_case_name] for alg in ALGORITHMS}
    
    save_publication_plot(
        history=None,
        filename="comparison_all_algorithms.png",
        is_comparison=True,
        all_histories=overall_histories,
        theoretical_max=SNR_theorem,
        labels={alg: f"{alg} (N={test_cases[1]['params']['pop_size']})" for alg in ALGORITHMS},
        title="Algorithm Comparison - Medium Population Size"
    )
    
    print("="*95)
    print(" "*35 + "✅ ALL EXPERIMENTS COMPLETED!")
    print("="*95)


if __name__ == "__main__":
    main()