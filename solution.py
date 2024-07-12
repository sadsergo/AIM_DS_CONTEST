import numpy as np
from itertools import combinations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


"""
    Стандартный алгоритм:
        1. Берем комбинацию из cols столбцов, cols от 2 до n, т.к. рассмотрение 1 столбца не подходит под условие о поднаборах
        2. Идем по каждой строке образованной подматрицы, по очереди меняем k-ый элемент на противоположный, k от 0 до длины строки, чтобы найти строку, которой нет в подматрице
        3. Если этой строки нет в подматрице, то заносим её в множество использованных строк, так как в другой раз может встретиться такая же строка, тогда будем учитывать результат о её принадлежности к закономерности дважды
        4. Надо проверить, что любое её разбиение на группы элементов от 1 до (длины строки - 1) есть в таком же разбиении подматрицы
        5. Если для такой строки эти 2 условия выполнены, то это закономерность
        6. Обновляем счетчик
        7. И так очень много много раз
        8. Возвращаем счетчик 
        9. .....очень долго (((
"""


def number_of_patterns(matrix: np.ndarray) -> int:
    num = 0
    n, m = matrix.shape

    for cols in range(1, 2**m):
        bin_list = np.fromstring(bin(cols)[2:], dtype=np.int8) - 48
        bin_cols = np.where(
            np.pad(bin_list, pad_width=(m - bin_list.shape[0], 0), mode="constant") == 1
        )[0]

        submatrix = matrix[:, bin_cols]
        _, sub_m = submatrix.shape

        #   Если подматрица состоит из одного столбца, то если столбец из одинаковых значений, то это закономерность
        #   Если из нескольких, то обрабатываем как написано в начале файла
        
        if sub_m == 1:
            num += np.unique(submatrix).shape[0] == 1
            continue

        used_rows = set()

        for i in range(n):
            for k in range(sub_m):
                row = submatrix[i, :].copy()
                row[k] ^= 1

                if tuple(row) in used_rows:
                    continue

                used_rows.add(tuple(row))

                if not np.any(np.all(submatrix == row, axis=1)):
                    has_pattern = True

                    for cols_count in range(1, sub_m):
                        for comb_cols in combinations(np.arange(0, sub_m), cols_count):
                            if not np.any(
                                np.all(
                                    submatrix[:, comb_cols] == row[np.array(comb_cols)],
                                    axis=1,
                                )
                            ):
                                has_pattern = False
                                break

                        if not has_pattern:
                            break

                    if has_pattern:
                        num += 1

    return num
