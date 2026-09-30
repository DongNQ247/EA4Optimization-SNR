import os
import numpy as np
import matplotlib.pyplot as plt

from population import Population
from problem    import Problem
from individual import Individual

def genetic_algorithm(
    problem: Problem,
    pop_size: int = 20,
    max_iter: int = 1500,
    pc: float = 0.5,
    pm: float = 0.5,
    patience: int = 50,
    history: bool = False,
    # --- MỚI: Thêm tham số để định danh Case và thư mục lưu ---
    case_name: str = "default",
    output_dir: str = "images" 
):
    population = Population(pop_size)
    population.gen_pop(problem)

    best_indiv   = population.best_individual()
    best_fitness = best_indiv.fitness
    no_improve   = 0

    fitness_history = []

    for i in range(max_iter):
        # Lưu lịch sử fitness
        fitness_history.append(best_fitness)

        children = []
        while len(children) < pop_size:
            p1, p2 = population.selection()
            if np.random.rand() < pc:
                c1, c2 = p1.crossover(p2)

                if np.random.rand() < pm:
                    c1.mutation()
                if np.random.rand() < pm:
                    c2.mutation()

                c1.cal_fitness(problem)
                c2.cal_fitness(problem)
                children.extend([c1, c2])

        population.list_indi.extend(children)
        population.survival_selection()

        current_best = population.best_individual()
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_indiv   = current_best
            no_improve   = 0  
        else:
            no_improve  += 1

        if no_improve >= patience:
            # Nếu dừng sớm, điền nốt giá trị hiện tại vào history cho biểu đồ đẹp
            # (Tùy chọn: bỏ dòng dưới nếu muốn biểu đồ cắt cụt tại điểm dừng)
            fitness_history.extend([best_fitness] * (max_iter - i - 1))
            break

        best_indiv = population.best_individual()

    # --- NÂNG CẤP: Phần vẽ biểu đồ từng Case ---
    if history:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(fitness_history, linewidth=1.5, color='blue')
        
        # Thêm thông tin chi tiết vào biểu đồ
        plt.xlabel("Generation")
        plt.ylabel("SNR (fitness)")
        
        # Tiêu đề động bao gồm thông số
        title_text = f"GA Optimization: {case_name}\n(N={pop_size}, Pc={pc}, Pm={pm})"
        plt.title(title_text, fontsize=10, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Lưu file với tên động: result_case1.png, result_case2.png...
        save_path = os.path.join(output_dir, f"result_{case_name.lower().replace(' ', '')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"-> Saved plot for {case_name} at {save_path}")

    # Trả về thêm fitness_history để main.py vẽ biểu đồ tổng hợp
    return best_indiv.decode(problem), fitness_history

def de(
    problem: Problem,
    pop_size: int = 20,
    max_iter: int = 1500,
    F: float = 0.8,        # Mutation factor
    CR: float = 0.9,       # Crossover rate
    patience: int = 50,
    history: bool = False,
    case_name: str = "default",
    output_dir: str = "images"
):
    """Differential Evolution Algorithm"""
    population = Population(pop_size)
    population.gen_pop(problem)

    best_indiv = population.best_individual()
    best_fitness = best_indiv.fitness
    no_improve = 0

    fitness_history = []

    for i in range(max_iter):
        fitness_history.append(best_fitness)

        # Tạo thế hệ mới
        new_population = []
        
        for target_idx in range(pop_size):
            target = population.list_indi[target_idx]
            
            # Chọn 3 cá thể ngẫu nhiên khác target
            candidates = [idx for idx in range(pop_size) if idx != target_idx]
            idx1, idx2, idx3 = np.random.choice(candidates, 3, replace=False)
            
            x1 = population.list_indi[idx1]
            x2 = population.list_indi[idx2]
            x3 = population.list_indi[idx3]
            
            # Mutation
            mutant = Individual()
            mutant.de_mutation(x1, x2, x3, F)
            
            # Crossover
            trial = mutant.de_crossover(target, CR)
            trial.cal_fitness(problem)
            
            # Selection
            if trial.fitness > target.fitness:
                new_population.append(trial)
            else:
                new_population.append(target)
        
        population.list_indi = new_population
        
        # Cập nhật best
        current_best = population.best_individual()
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_indiv = current_best
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            fitness_history.extend([best_fitness] * (max_iter - i - 1))
            break

    # Vẽ biểu đồ
    if history:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(fitness_history, linewidth=1.5, color='red')
        
        plt.xlabel("Generation")
        plt.ylabel("SNR (fitness)")
        
        title_text = f"DE Optimization: {case_name}\n(N={pop_size}, F={F}, CR={CR})"
        plt.title(title_text, fontsize=10, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"result_{case_name.lower().replace(' ', '')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"-> Saved plot for {case_name} at {save_path}")

    return best_indiv.decode(problem), fitness_history


def pso(
    problem: Problem,
    pop_size: int = 20,
    max_iter: int = 1500,
    w: float = 0.7,        # Inertia weight
    c1: float = 1.5,       # Cognitive parameter
    c2: float = 1.5,       # Social parameter
    patience: int = 50,
    history: bool = False,
    case_name: str = "default",
    output_dir: str = "images"
):
    """Particle Swarm Optimization Algorithm"""
    population = Population(pop_size)
    population.gen_pop(problem)

    gbest = population.best_individual()  # Global best
    best_fitness = gbest.fitness
    no_improve = 0

    fitness_history = []

    for i in range(max_iter):
        fitness_history.append(best_fitness)

        # Cập nhật từng particle
        for particle in population.list_indi:
            particle.pso_update(gbest, w, c1, c2)
            particle.cal_fitness(problem)
            
        # Cập nhật global best
        current_best = population.best_individual()
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            gbest = current_best
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            fitness_history.extend([best_fitness] * (max_iter - i - 1))
            break

    # Vẽ biểu đồ
    if history:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(fitness_history, linewidth=1.5, color='green')
        
        plt.xlabel("Generation")
        plt.ylabel("SNR (fitness)")
        
        title_text = f"PSO Optimization: {case_name}\n(N={pop_size}, w={w}, c1={c1}, c2={c2})"
        plt.title(title_text, fontsize=10, fontweight='bold')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"result_{case_name.lower().replace(' ', '')}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"-> Saved plot for {case_name} at {save_path}")

    return gbest.decode(problem), fitness_history

