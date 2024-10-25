import numpy as np
from scipy.linalg import svd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from memory_profiler import memory_usage

# Завантаження початкових зображень
def load_image_as_matrix(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {filepath}")
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0.8)
    img_equalized = cv2.equalizeHist(img_blurred)
    img_matrix = img_equalized.astype(np.float64) / 255.0

    return img_matrix

def preprocess_images(X, Y):
    if X.shape[0] < Y.shape[0]:
        padding = np.ones((1, X.shape[1]))
        X = np.vstack([X, padding])
    return X, Y

# Методи Мура-Пенроуз для обчислення псевдооберненої матриці
def pseudo_inverse_moore_penrose_1(X, tolerance=1e-10, max_iterations=1000):
    # Початкова оцінка псевдооберненої матриці
    X_T = X.T
    X_frobenius_norm = np.linalg.norm(X, 'fro')
    X_pinv = X_T / X_frobenius_norm**2

    # Ітераційне уточнення
    for iteration in range(max_iterations):
        previous_X_pinv = X_pinv
        X_pinv = X_pinv + X_pinv @ (np.eye(X.shape[0]) - X @ X_pinv)

        # Перевірка збіжності
        if np.linalg.norm(X_pinv - previous_X_pinv) < tolerance:
            break
    else:
        print("Max iterations reached without full convergence")

    return X_pinv

def pseudo_inverse_moore_penrose_2(X, threshold=1e-10):
    # Обчислюємо власні значення та вектори
    eigvals, eigvecs = np.linalg.eigh(np.dot(X.T, X))

    # Інвертуємо тільки ненульові (вище порогу) власні значення
    inv_eigvals = np.array([1 / ev if ev > threshold else 0 for ev in eigvals])

    # Псевдообернена матриця
    X_pinv = eigvecs @ np.diag(inv_eigvals) @ eigvecs.T @ X.T
    return X_pinv

# Метод сингулярного розкладу для обчислення псевдооберненої матриці
def pseudo_inverse_svd(X, threshold=1e-10):
    U, S, Vt = svd(X, full_matrices=False)
    S_inv = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] > threshold:
            S_inv[i] = 1 / S[i]

    X_pinv = Vt.T @ np.diag(S_inv) @ U.T
    return X_pinv

def Z(A, A_pseudo_inverse):
    return np.eye(A_pseudo_inverse.shape[0]) - A_pseudo_inverse @ A

# Формула Гревіля для обчислення псевдооберненої матриці
def greville_pseudo_inverse(A):
    is_swap = False

    # Якщо матриця має більше рядків ніж стовпців, потрібно транспонувати
    if A.shape[0] > A.shape[1]:
        is_swap = True
        A = A.T

    # Початковий вектор (перший рядок, транспонований)
    current_vector = A[0, :].reshape(-1, 1)

    # Ініціалізація псевдооберненої матриці
    vector_scalar = current_vector.T @ current_vector
    if vector_scalar == 0:
        A_pseudo_inverse = current_vector
    else:
        A_pseudo_inverse = current_vector / vector_scalar

    # Ініціалізуємо A_i як перший рядок
    A_i = current_vector.T

    # Основний цикл для решти рядків матриці A
    for i in range(1, A.shape[0]):
        current_vector = A[i, :].reshape(-1, 1)  # Поточний рядок
        Z_A = Z(A_i, A_pseudo_inverse)  # Обчислюємо Z_A

        # Додаємо новий рядок до A_i
        A_i = np.vstack([A_i, current_vector.T])

        # Обчислення знаменника
        denom_Z = current_vector.T @ Z_A @ current_vector

        # Оновлення псевдооберненої матриці залежно від знаменника
        if denom_Z > 0:
            A_pseudo_inverse = np.hstack([
                A_pseudo_inverse - (Z_A @ current_vector @ current_vector.T @ A_pseudo_inverse) / denom_Z,
                (Z_A @ current_vector) / denom_Z
            ])
        else:
            R_A = A_pseudo_inverse @ A_pseudo_inverse.T
            denom_R = 1 + current_vector.T @ R_A @ current_vector
            A_pseudo_inverse = np.hstack([
                A_pseudo_inverse - (R_A @ current_vector @ current_vector.T @ A_pseudo_inverse) / denom_R,
                (R_A @ current_vector) / denom_R
            ])

    # Якщо ми транспонували матрицю на початку, повертаємо її назад
    if is_swap:
        A_pseudo_inverse = A_pseudo_inverse.T

    return A_pseudo_inverse

# Перевірка виконання характеристичної властивості псевдооберненої матриці
def check_pseudo_inverse_properties(A, A_pseudo_inverse):
    def check_property(property_value, threshold=1e-6):
        return abs(property_value) < threshold

    # Перевірка AA+A = A
    A_A_pinv_A = A @ A_pseudo_inverse @ A
    property_1_mse = calculate_mse(A, A_A_pinv_A)
    property_1_check = check_property(property_1_mse)

    # Перевірка A+AA+ = A+
    A_pinv_A_A_pinv = A_pseudo_inverse @ A @ A_pseudo_inverse
    property_2_mse = calculate_mse(A_pseudo_inverse, A_pinv_A_A_pinv)
    property_2_check = check_property(property_2_mse)

    # Перевірка, чи є AA+ симетричною
    AA_pinv = A @ A_pseudo_inverse
    property_3_mse = calculate_mse(AA_pinv, AA_pinv.T)
    property_3_check = check_property(property_3_mse)

    # Перевірка, чи є A+A симетричною
    A_pinv_A = A_pseudo_inverse @ A
    property_4_mse = calculate_mse(A_pinv_A, A_pinv_A.T)
    property_4_check = check_property(property_4_mse)

    return {
        "AA+A = A": property_1_check,
        "A+AA+ = A+": property_2_check,
        "AA+ is symmetric": property_3_check,
        "A+A is symmetric": property_4_check,
    }

# Обчислення лінійного оператора та трансформування зображення
def compute_linear_operator_and_transform(X, Y, method='mp1'):
    if method in ['mp1', 'mp2', 'svd', 'greville']:
        X, Y = preprocess_images(X, Y)

    # Обчислення псевдооберненої матриці відповідно до методу
    if method == 'mp1':
        X_pinv = pseudo_inverse_moore_penrose_1(X)
    elif method == 'mp2':
        X_pinv = pseudo_inverse_moore_penrose_2(X)
    elif method == 'svd':
        X_pinv = pseudo_inverse_svd(X)
    elif method == 'greville':
        X_pinv = greville_pseudo_inverse(X)
    else:
        raise ValueError("Unknown method")

    # обчислення лінійного оператора
    A_operator = Y @ X_pinv
    Y_transformed = A_operator @ X
    return Y_transformed


def save_image_as_bmp(image_array, filename):
    img = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    if len(img.shape) == 2:
        img_pil = Image.fromarray(img, mode='L')
    else:
        img_pil = Image.fromarray(img)
    img_pil.save(filename, format='BMP')

# Порівняння псевдообернених матриць
def compare_pseudo_inverses(Y, Y_transformed_dict):
    comparisons = {}

    # Порівняння кожного трансформованого зображення з початковим Y
    for method_name, Y_transformed in Y_transformed_dict.items():
        difference = np.linalg.norm(Y - Y_transformed)
        comparisons[f"{method_name} vs Y"] = difference

    return comparisons

# Обчислення середньоквадратичної похибки
def calculate_mse(original, transformed):
    # Обчислення квадратів різниць
    squared_diffs = (original - transformed) ** 2
    # Сума квадратів різниць
    total_squared_diff = np.sum(squared_diffs)
    # Повертаємо середнє значення
    mse = total_squared_diff / original.size
    return mse

# Виведення результатів
def visualize_results(X, Y, Y_transformed_dict, time_results, memory_results):
    plt.figure(figsize=(5, 5))
    plt.imshow(X[:-1, :], cmap='gray')
    plt.title('Input Image (X)')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(Y, cmap='gray')
    plt.title('Output Image (Y)')
    plt.axis('off')
    plt.show()

    # Виведення трансформованих зображень
    for method_name, Y_transformed in Y_transformed_dict.items():
        plt.figure(figsize=(5, 5))
        plt.imshow(Y_transformed, cmap='gray')
        plt.title(f'Transformed Image (Y\') using {method_name}')
        plt.axis('off')
        plt.show()

        # Збереження трансформованих зображень
        save_image_as_bmp(Y_transformed, f'transformed_{method_name}.bmp')

    # Графік порівняння часу
    plt.figure(figsize=(10, 5))
    for method_name, times in time_results.items():
        plt.plot(range(len(times)), times, marker='o', label=method_name)

    plt.title('Execution Time for Each Method')
    plt.xlabel('Run')
    plt.ylabel('Time (s)')
    plt.xticks(range(len(times)), [f'Run {i + 1}' for i in range(len(times))])
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Графік використання пам'яті
    plt.figure(figsize=(10, 5))
    for method_name, memories in memory_results.items():
        plt.plot(range(len(memories)), memories, marker='o', label=method_name)

    plt.title('Memory Usage for Each Method')
    plt.xlabel('Run')
    plt.ylabel('Memory Used (MB)')
    plt.xticks(range(len(memories)), [f'Run {i + 1}' for i in range(len(memories))])
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Гістограма порівняння яскравості
    plt.figure(figsize=(10, 5))
    for method_name, Y_transformed in Y_transformed_dict.items():

        hist, bins = np.histogram(Y_transformed, bins=256, range=(0, 1))
        plt.plot(bins[:-1], hist, label=f'{method_name} histogram')

    plt.title('Histogram Comparison of Transformed Images')
    plt.xlabel('Brightness Level')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pseudo_inverse_comparisons(comparisons):
    plt.figure(figsize=(10, 5))
    names = list(comparisons.keys())
    differences = list(comparisons.values())

    plt.bar(names, differences, color='skyblue')
    plt.ylabel('Difference in Norm')
    plt.title('Differences Between Pseudo-Inverse Matrices')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":

    X = load_image_as_matrix('x1.bmp')
    Y = load_image_as_matrix('y9.bmp')

    Y_transformed_dict = {}
    time_results = {method: [] for method in ['mp1', 'mp2', 'svd', 'greville']}
    memory_results = {method: [] for method in ['mp1', 'mp2', 'svd', 'greville']}
    pinv_matrices = {}
    mse_results = {method: [] for method in ['mp1', 'mp2', 'svd', 'greville']}

    methods = ['mp1', 'mp2', 'svd', 'greville']
    for method in methods:
        for run in range(5):
            start_time = time.time()
            mem_usage = memory_usage((compute_linear_operator_and_transform, (X, Y, method)))
            end_time = time.time()

            elapsed_time = end_time - start_time
            max_memory = max(mem_usage) - mem_usage[0]

            time_results[method].append(elapsed_time)
            memory_results[method].append(max_memory)
            Y_transformed = compute_linear_operator_and_transform(X, Y, method)
            Y_transformed_dict[method] = Y_transformed

            if method == 'mp1':
                pinv_matrices['mp1'] = pseudo_inverse_moore_penrose_1(X)
                properties_mse = check_pseudo_inverse_properties(X, pinv_matrices['mp1'])
            elif method == 'mp2':
                pinv_matrices['mp2'] = pseudo_inverse_moore_penrose_2(X)
                properties_mse = check_pseudo_inverse_properties(X, pinv_matrices['mp2'])
            elif method == 'svd':
                pinv_matrices['svd'] = pseudo_inverse_svd(X)
                properties_mse = check_pseudo_inverse_properties(X, pinv_matrices['svd'])
            elif method == 'greville':
                pinv_matrices['greville'] = greville_pseudo_inverse(X)
                properties_mse = check_pseudo_inverse_properties(X, pinv_matrices['greville'])

            mse = calculate_mse(Y, Y_transformed)
            mse_results[method].append(mse)

            properties_results = check_pseudo_inverse_properties(X, pinv_matrices[method])
            print(
                f"Method: {method}, Run: {run + 1}, Time: {elapsed_time:.4f} s, Memory: {max_memory:.4f} MB, Середньоквадратична похибка: {mse:.6f}")
            for property_name, result in properties_results.items():
                print(f"{property_name}: {'True' if result else 'False'}")

    comparisons = compare_pseudo_inverses(Y, Y_transformed_dict)
    for comparison, difference in comparisons.items():
        print(f"Difference between {comparison}: {difference:.4f}")

    plot_pseudo_inverse_comparisons(comparisons)

    visualize_results(X, Y, Y_transformed_dict, time_results, memory_results)