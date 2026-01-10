import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. Параметры модели ---

# Размер сетки и время
Nx, Ny = 100, 100   # Размер решетки
T_steps = 50       # Количество шагов времени (клеточных циклов)
PDE_steps = 30      # Шагов диффузии на один шаг клеточного автомата
dt = 0.1            # Шаг времени для диффузии
dx = 1.0            # Шаг пространства

# Коэффициенты диффузии и среды
D0 = 0.5            # Базовая диффузия
alpha = 2.0         # Коэффициент "запирания" среды клетками (блокировка)
k_cons = 0.05       # Скорость потребления кислорода клетками
source_rate = 0.05  # Источник кислорода (S) - пополнение из сосудов

# Параметры вероятностей (калибровка для визуальной наглядности)
# Мутация
P_base_mut = 0.001
P_max_mut = 0.05
b_mut = 5.0

# Пролиферация (деление)
P_max_prolif = 0.15
g_prolif = 5.0

# Смерть (апоптоз/некроз)
P_base_death = 0.01
P_max_death = 0.1
d_death = 10.0

# --- 2. Инициализация полей ---

# Поле клеток: 0 - пусто, 1 - нормальная клетка, 2 - мутировавшая (для наглядности P_mut)
cells = np.zeros((Nx, Ny))
center_x, center_y = Nx // 2, Ny // 2
cells[center_x, center_y] = 1  # Начальная клетка
cells[center_x+1, center_y] = 1
cells[center_x, center_y+1] = 1

# Поле кислорода (концентрация c)
# Начинаем с полного насыщения, границы будут источником
oxygen = np.ones((Nx, Ny))

# История популяции для графика
history_cells = []
history_mutations = []

# --- 3. Вспомогательные функции ---

def get_probabilities(c):
    """Вычисляет вероятности событий в зависимости от локального кислорода c."""
    # Вероятность мутации: Pmut(c) = Pbase + Pmax*exp(-bc)
    p_mut = P_base_mut + P_max_mut * np.exp(-b_mut * c)
    p_mut = np.clip(p_mut, 0, 1)
    
    # Вероятность пролиферации: Pprolif(c) = Pmax(1 - exp(-gc))
    p_prolif = P_max_prolif * (1 - np.exp(-g_prolif * c))
    p_prolif = np.clip(p_prolif, 0, 1)
    
    # Вероятность смерти: Pdeath(c) = Pbase + Pmax*exp(-dc)
    p_death = P_base_death + P_max_death * np.exp(-d_death * c)
    p_death = np.clip(p_death, 0, 1)
    
    return p_mut, p_prolif, p_death

def diffusion_step(c, cells_grid):
    """Решает уравнение диффузии на один шаг dt."""
    # Плотность rho (0 или 1) для уравнения. Считаем мутантов (2) тоже за 1.
    rho = (cells_grid > 0).astype(float)
    
    # D(p) = D0 * exp(-alpha * rho)
    D = D0 * np.exp(-alpha * rho)
    
    # Лапласиан с переменным коэффициентом (упрощенная схема 5 точек)
    # d/dx (D dC/dx) ~ (D[i+1]*(C[i+1]-C[i]) - D[i-1]*(C[i]-C[i-1])) / dx^2
    # Для скорости используем np.roll
    
    c_up = np.roll(c, -1, axis=0)
    c_down = np.roll(c, 1, axis=0)
    c_left = np.roll(c, -1, axis=1)
    c_right = np.roll(c, 1, axis=1)
    
    # Дискретная диффузия
    laplacian = (c_up + c_down + c_left + c_right - 4*c)
    
    # Уравнение: dc/dt = div(D grad c) + S - k*rho*c
    # Упрощаем div(D grad c) до D * laplacian(c) для стабильности в простой модели,
    # хотя строго говоря нужно (grad D * grad c + D * laplacian c).
    # Для клеточных автоматов часто достаточно D * laplacian.
    
    diffusion_term = D * laplacian / (dx**2)
    reaction_term = -k_cons * rho * c
    
    # Источник S (например, постоянный приток на границах и слабый фон)
    source = np.zeros_like(c)
    # Граничные условия (Dirichlet) - кислород поступает с краев
    source[0, :] = 0.1
    source[-1, :] = 0.1
    source[:, 0] = 0.1
    source[:, -1] = 0.1
    
    c_new = c + dt * (diffusion_term + reaction_term + source)
    
    # Фиксация границ (постоянный приток из "сосудов" вокруг ткани)
    c_new[0, :] = 1.0
    c_new[-1, :] = 1.0
    c_new[:, 0] = 1.0
    c_new[:, -1] = 1.0
    
    return np.clip(c_new, 0, 1.0)

# --- 4. Основной цикл симуляции ---

print("Запуск симуляции...")

for t in tqdm(range(T_steps)):
    # A. Шаг PDE (Диффузия кислорода)
    # Поле кислорода обновляется быстрее, чем клетки делятся
    for _ in range(PDE_steps):
        oxygen = diffusion_step(oxygen, cells)
    
    # B. Шаг CA (Клеточный автомат)
    # Создаем копию, чтобы изменения применялись синхронно
    new_cells = cells.copy()
    
    # Получаем координаты живых клеток
    rows, cols = np.where(cells > 0)
    
    # Перемешиваем порядок обхода, чтобы избежать артефактов направления
    indices = list(zip(rows, cols))
    np.random.shuffle(indices)
    
    for r, c in indices:
        cell_type = cells[r, c] # 1 - обычная, 2 - мутант
        local_o2 = oxygen[r, c]
        
        p_mut, p_prolif, p_death = get_probabilities(local_o2)
        
        # 1. Проверка на смерть
        if np.random.random() < p_death:
            new_cells[r, c] = 0
            continue # Клетка умерла, дальше не обрабатываем
            
        # 2. Проверка на мутацию (если клетка еще не мутант)
        if cell_type == 1 and np.random.random() < p_mut:
            new_cells[r, c] = 2 # Стала мутантом
            cell_type = 2
            
        # 3. Пролиферация (деление)
        # Мутанты могут иметь преимущество, здесь используем общую формулу,
        # но для мутантов можно было бы увеличить P_max_prolif
        current_p_prolif = p_prolif
        if cell_type == 2:
            current_p_prolif *= 1.2 # Бонус мутантам
            
        if np.random.random() < current_p_prolif:
            # Ищем свободного соседа (Moore neighborhood - 8 соседей)
            neighbors = [
                (r-1, c), (r+1, c), (r, c-1), (r, c+1),
                (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
            ]
            empty_spots = []
            for nr, nc in neighbors:
                if 0 <= nr < Nx and 0 <= nc < Ny and new_cells[nr, nc] == 0:
                    empty_spots.append((nr, nc))
            
            if empty_spots:
                # Выбираем случайное место для новой клетки
                nr, nc = empty_spots[np.random.randint(len(empty_spots))]
                new_cells[nr, nc] = cell_type # Дочерняя клетка того же типа
                
    cells = new_cells
    
    # Сохраняем статистику
    n_normal = np.sum(cells == 1)
    n_mutant = np.sum(cells == 2)
    history_cells.append(n_normal)
    history_mutations.append(n_mutant)

# --- 5. Визуализация результатов ---

plt.figure(figsize=(18, 5))

# График 1: Динамика популяции
plt.subplot(1, 3, 1)
plt.plot(history_cells, label='Нормальные клетки', color='blue')
plt.plot(history_mutations, label='Мутировавшие клетки', color='red')
plt.plot(np.array(history_cells) + np.array(history_mutations), label='Всего', color='black', linestyle='--')
plt.title('Динамика роста опухоли')
plt.xlabel('Время (шаги CA)')
plt.ylabel('Количество клеток')
plt.legend()
plt.grid(True)

# График 2: Heatmap кислорода
plt.subplot(1, 3, 2)
sns.heatmap(oxygen, cmap='viridis', vmin=0, vmax=1, square=True, cbar_kws={'label': 'Концентрация O2'})
plt.title('Концентрация кислорода (c)\nТемнее = Гипоксия')
plt.axis('off')

# График 3: Heatmap клеток
plt.subplot(1, 3, 3)
# Создаем кастомную карту цветов: 0-белый, 1-синий, 2-красный
from matplotlib.colors import ListedColormap
cmap_cells = ListedColormap(['white', '#4a90e2', '#e74c3c'])
sns.heatmap(cells, cmap=cmap_cells, square=True, cbar=False)
plt.title('Распределение клеток\nСиние: Норма, Красные: Мутанты')
plt.axis('off')

plt.tight_layout()
plt.show()