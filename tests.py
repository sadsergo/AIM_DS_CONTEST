import numpy as np
from solution import number_of_patterns


def calc_identity_patterns(n : int) -> int:
    return 1 + n * (n - 1) / 2


def calc_n1_patterns(matrix : np.array) -> int:
    return int(np.unique(matrix).shape[0] == 1)


def calc_exp_patterns(matrix : np.array) -> int:
    return int(np.array_equal(np.unique(matrix, axis=0), matrix))


def calc_2n_patterns(matrix: np.array) -> int:
    if len(np.where((matrix==[0] * matrix.shape[1]).all(axis=1))[0]) > 0 and len(np.where((matrix==[1] * matrix.shape[1]).all(axis=1))[0]) > 0:
        return matrix.shape[1] * (matrix.shape[1] - 1)

    return 0


def test_identity():
    for n in range(1, 6):
        identity = np.identity(n=n).astype(np.int64)

        test_num = calc_identity_patterns(n)
        cur_num = number_of_patterns(identity)

        if test_num != cur_num:
            print("IDENTITY TEST NOT PASSED")
            print(f"N = {n}, NEED NUM: {test_num}, CURRENT IS {cur_num}")
        else:
            print(f"N = {n}, TEST IS PASSED")
    
    print()


def test_n1():
    matrix1 = np.array([[0], [0], [0], [1]])
    matrix2 = np.zeros(shape=(100, 1))
    matrix3 = np.array([[1], [1], [1], [1], [1], [1]])

    for test_id, matrix in enumerate([matrix1, matrix2, matrix3]):
        test_num = calc_n1_patterns(matrix)
        cur_num = number_of_patterns(matrix)

        if test_num != cur_num:
            print("TEST NOT PASSED")
            print(f"TEST_{test_id}, NEED NUM: {test_num}, CURRENT IS {cur_num}")
        else:
            print(f"TEST_{test_id}, TEST IS PASSED")

    print()


def test_exp():
    matrix1 = np.array([[0, 0, 0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]])
    matrix2 = np.array([[0, 0], [0, 1], [1, 0]])
    matrix3 = np.array([[0, 0, 0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,1]])

    for test_id, matrix in enumerate([matrix1, matrix2, matrix3]):
        test_num = calc_exp_patterns(matrix)
        cur_num = number_of_patterns(matrix)

        if test_num != cur_num:
            print("TEST NOT PASSED")
            print(f"TEST_{test_id}, NEED NUM: {test_num}, CURRENT IS {cur_num}")
        else:
            print(f"TEST_{test_id}, TEST IS PASSED")
    
    print()


def test_2n():
    matrix1 = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    matrix2 = np.array([[0] * 10, [1] * 10])

    for test_id, matrix in enumerate([matrix1, matrix2]):
        test_num = calc_2n_patterns(matrix)
        cur_num = number_of_patterns(matrix)

        if test_num != cur_num:
            print("TEST NOT PASSED")
            print(f"TEST_{test_id}, NEED NUM: {test_num}, CURRENT IS {cur_num}")
        else:
            print(f"TEST_{test_id}, TEST IS PASSED")
    
    print()


def test_example1():
    print('TEST ON EXAMPLE 1')

    matrix = np.array(
        [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 0, 1, 1, 1]]
    )

    test_num = 13
    cur_num = number_of_patterns(matrix)

    if test_num != cur_num:
        print("TEST NOT PASSED")
        print(f"NEED NUM: {test_num}, CURRENT IS {cur_num}")
    else:
        print(f"TEST IS PASSED")

    print()


def test_example2():
    print("TEST ON EXAMPLE 2")

    matrix = np.array([[0, 1, 1, 0],
                   [1, 1, 1, 1],
                   [0, 0, 0, 1]])

    test_num = 9
    cur_num = number_of_patterns(matrix)

    if test_num != cur_num:
        print("TEST NOT PASSED")
        print(f"NEED NUM: {test_num}, CURRENT IS {cur_num}")
    else:
        print(f"TEST IS PASSED")

    print()


def run(test_id : int):
    if test_id == 1:
        print("TEST IDENTITY")

        test_identity()
    elif test_id == 2:
        print("TEST N1")

        test_n1()
    elif test_id == 3:
        print("TEST 2^N")

        test_exp()
    elif test_id == 4:
        print('TEST 2N')

        test_2n()


def run_all_tests():
    for test_id in range(1, 5):
        run(test_id)

    test_example1()
    test_example2()
