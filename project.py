import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import os

# --- 1. Параметры модели ---

Nx, Ny = 200, 200
T_steps = 100
PDE_steps = 20
dt = 0.1

D0 = 0.01
alpha = 2.0
k_cons = 0.05
source_rate = 0.1

# Параметры HIF
k_h = 0.2           # Производство HIF
lambda_h = 0.1      # Распад
g_hif = 8.0         # Чувствительность производства к гипоксии

# --- НОВЫЕ ПАРАМЕТРЫ МУТАГЕНЕЗА ---
P_base_mut = 0.001

# Вес фактора "Кислород" (Мгновенный стресс / ROS)
P_ros_factor = 0.05
b_ros = 5.0  # Насколько резко растет мутация при падении O2

# Вес фактора "HIF" (Подавление репарации)
P_hif_factor = 0.1 

# Пролиферация и смерть
P_max_prolif = 0.15
g_prolif = 5.0
P_base_death = 0.005
P_max_death = 0.15
d_death = 10.0

# --- 2. Инициализация ---

cells = np.zeros((Nx, Ny))
cx, cy = Nx // 2, Ny // 2
r_start = 2
for x in range(cx-r_start, cx+r_start+1):
    for y in range(cy-r_start, cy+r_start+1):
        if (x-cx)**2 + (y-cy)**2 <= r_start**2:
            cells[x, y] = 1

oxygen = np.ones((Nx, Ny))
hif_field = np.zeros((Nx, Ny))

history_cells = []
history_mutants = []

# --- 3. Функции ---

def update_hif(h, c, cells_mask):
    """
    h'(x, t) = k_h * (1 - exp(-g(1-c))) - lambda_h * h
    """
    production = k_h * (1 - np.exp(-g_hif * (1 - c)))
    decay = lambda_h * h
    h_new = h + dt * (production - decay)
    return np.clip(h_new, 0, None)

def get_probabilities(c, h):
    """
    Рассчитывает вероятности с учетом ДВУХ факторов для мутации.
    """
    # 1. Мутация: P(c, h)
    # Слагаемое 1: Прямой эффект гипоксии (ROS), растет при падении c
    term_ros = P_ros_factor * np.exp(-b_ros * c)
    
    # Слагаемое 2: Эффект HIF (ошибки репарации), растет линейно от h
    term_hif = P_hif_factor * h
    
    p_mut = P_base_mut + term_ros + term_hif
    p_mut = np.clip(p_mut, 0, 1)
    
    # 2. Пролиферация (энергия от O2)
    p_prolif = P_max_prolif * (1 - np.exp(-g_prolif * c))
    p_prolif = np.clip(p_prolif, 0, 1)
    
    # 3. Смерть (критическая гипоксия)
    p_death = P_base_death + P_max_death * np.exp(-d_death * c)
    p_death = np.clip(p_death, 0, 1)
    
    return p_mut, p_prolif, p_death

def diffusion_step_o2(c, cells_grid):
    rho = (cells_grid > 0).astype(float)
    D = D0 * np.exp(-alpha * rho)
    
    c_up = np.roll(c, -1, axis=0)
    c_down = np.roll(c, 1, axis=0)
    c_left = np.roll(c, -1, axis=1)
    c_right = np.roll(c, 1, axis=1)
    
    laplacian = (c_up + c_down + c_left + c_right - 4*c)
    
    bc_mask = np.zeros_like(c)
    bc_mask[0,:]=1; bc_mask[-1,:]=1; bc_mask[:,0]=1; bc_mask[:,-1]=1
    
    dc_dt = (D * laplacian) - (k_cons * rho * c) + (source_rate * bc_mask * (1-c))
    return np.clip(c + dt * dc_dt, 0, 1.0)

# --- 4. Основной цикл ---

print("Запуск модели: Гипоксия + HIF зависимый мутагенез...")

for t in tqdm(range(T_steps)):
    # 1. Поле O2
    for _ in range(PDE_steps):
        oxygen = diffusion_step_o2(oxygen, cells)
    
    # 2. Поле HIF (обновляем везде, но учитываем физику внутри клеток позже)
    hif_field = update_hif(hif_field, oxygen, cells)
    
    # 3. Клеточный автомат
    new_cells = cells.copy()
    rows, cols = np.where(cells > 0)
    indices = list(zip(rows, cols))
    np.random.shuffle(indices)
    
    for r, c_idx in indices:
        cell_type = cells[r, c_idx]
        loc_o2 = oxygen[r, c_idx]
        loc_hif = hif_field[r, c_idx]
        
        # Передаем ОБА параметра
        p_mut, p_prolif, p_death = get_probabilities(loc_o2, loc_hif)
        
        if np.random.random() < p_death:
            new_cells[r, c_idx] = 0
            continue
            
        # Логика мутации
        if cell_type == 1 and np.random.random() < p_mut:
            new_cells[r, c_idx] = 2
            cell_type = 2
            
        # Логика деления
        eff_prolif = p_prolif * (1.3 if cell_type == 2 else 1.0)
        
        if np.random.random() < eff_prolif:
            neighbors = [
                (r-1, c_idx), (r+1, c_idx), (r, c_idx-1), (r, c_idx+1),
                (r-1, c_idx-1), (r-1, c_idx+1), (r+1, c_idx-1), (r+1, c_idx+1)
            ]
            empty = [(nr, nc) for nr, nc in neighbors 
                     if 0 <= nr < Nx and 0 <= nc < Ny and new_cells[nr, nc] == 0]
            
            if empty:
                nr, nc = empty[np.random.randint(len(empty))]
                new_cells[nr, nc] = cell_type

    cells = new_cells
    history_cells.append(np.sum(cells == 1))
    history_mutants.append(np.sum(cells == 2))

# --- 5. Визуализация ---

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle(f"Гипоксия-индуцированный мутагенез (ROS + HIF)", fontsize=16)

# Динамика
axs[0, 0].plot(history_cells, label='Norm', c='blue')
axs[0, 0].plot(history_mutants, label='Mutant', c='red')
axs[0, 0].set_title("Популяции")
axs[0, 0].legend()
axs[0, 0].grid(True)

# O2
sns.heatmap(oxygen, ax=axs[0, 1], cmap='viridis', vmin=0, vmax=1, square=True)
axs[0, 1].set_title("Кислород (c)")
axs[0, 1].axis('off')

# HIF
sns.heatmap(hif_field, ax=axs[1, 0], cmap='magma', square=True)
axs[1, 0].set_title("HIF-1a (h)\nНакопленный фактор")
axs[1, 0].axis('off')

# Клетки
cmap_cells = ListedColormap(['white', '#4a90e2', '#e74c3c'])
sns.heatmap(cells, ax=axs[1, 1], cmap=cmap_cells, square=True, cbar=False)
axs[1, 1].set_title("Клетки")
axs[1, 1].axis('off')


filename = f'Figure-step{T_steps}-D0-{D0}.png'
folderpath = '/Users/musakair/Desktop/диффузия/test HIF ROS'
full_path = os.path.join(folderpath, filename)
plt.savefig(full_path, dpi=300, format='png')

plt.tight_layout()
plt.show()
