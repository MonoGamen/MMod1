import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


N, M = 3, 4
ITER_COUNT = 10000
Ai, Bj = [x for x in range(N)], [x for x in range(M)]


def getIndex(array):
    rand0_1 = np.random.random()
    index = len(array) - 1
    for x in array:
        if x >= rand0_1:
            index = array.index(x)
            break
    return index


def generateValue(Pij):
    sumEveryRow = [sum(x) for x in Pij]
    rowIntervals = [sum([sumEveryRow[t] for t in range(x + 1)]) for x in range(N)]
    rowIndex = getIndex(rowIntervals)

    normalizedRow = [x / sum(Pij[rowIndex]) for x in Pij[rowIndex]]
    columnIntervals = [sum([normalizedRow[t] for t in range(x + 1)]) for x in range(M)]
    colIndex = getIndex(columnIntervals)

    return (rowIndex, colIndex)


def getM(Pij, Ai, Bj):
    rowM = 0
    for i in range(N):
        for j in range(M):
            rowM += Ai[i] * Pij[i][j]

    colM = 0
    for j in range(M):
        for i in range(N):
            colM += Bj[j] * Pij[i][j]

    return (rowM, colM)


def showPlots(Pij, normalizedCountMatrix):
    theorProbabilityAi, theorProbabilityBj = np.sum(Pij, axis=1), np.sum(Pij, axis=0)
    empirProbabilityAi, empirProbabilityBj = np.sum(normalizedCountMatrix, axis=1), np.sum(normalizedCountMatrix,
                                                                                           axis=0)
    data = {'Теор. вер. A': theorProbabilityAi.tolist(), 'Эмпир. вер. A': empirProbabilityAi.tolist()}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.show()

    data = {'Теор. вер. B': theorProbabilityBj.tolist(), 'Эмпир. вер. B': empirProbabilityBj.tolist()}
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.show()

    sns.heatmap(normalizedCountMatrix, cmap='summer')
    plt.show()


def showCorrelation(normalizedCountMatrix, empirMAi, empirMBj, empirDAi, empirDBj):
    M_AB = 0
    for i in range(N):
        for j in range(M):
            M_AB += normalizedCountMatrix[i][j] * Ai[i] * Bj[j]
    covariation = M_AB - empirMAi * empirMBj

    correlation = covariation / np.sqrt(empirDAi * empirDBj)
    print(f'Коэффициент корреляции = {correlation}\n')


def chi2(Pij, normalizedCountMatrix):
    chi = ITER_COUNT * np.sum(np.square(normalizedCountMatrix - Pij) / Pij)
    print(f'Коэффициент согласия Пирсона = {chi}')
    ppf = stats.chi2.ppf(0.95, Pij.shape[0] * Pij.shape[1] - 1)
    print(chi, '<' if chi < ppf else '>', ppf)
    if chi < ppf:
        print(f'Эмпирическое распределение сходится к теоретическому\n')
    else:
        print(f'Эмпирическое распределение не сходится к теоретическому\n')


def intervalM(MA, DA, MB, DB):
    delta = stats.norm.ppf(0.975) * np.sqrt(DA) / np.sqrt(ITER_COUNT)
    print(f'Интервальная оценка мат. ожидания А: ({MA - delta}, {MA + delta})')
    delta = stats.norm.ppf(0.975) * np.sqrt(DB) / np.sqrt(ITER_COUNT)
    print(f'Интервальная оценка мат. ожидания B: ({MB - delta}, {MB + delta})')


def intervalD(DA, DB):
    leftDA = ITER_COUNT * DA / stats.chi2.isf(0.025, ITER_COUNT - 1)
    rightDA = ITER_COUNT * DA / stats.chi2.isf(0.975, ITER_COUNT - 1)
    print(f'Интервальная оценка дисперсии А: ({leftDA}, {rightDA})')
    leftDB = ITER_COUNT * DB / stats.chi2.isf(0.025, ITER_COUNT - 1)
    rightDB = ITER_COUNT * DB / stats.chi2.isf(0.975, ITER_COUNT - 1)
    print(f'Интервальная оценка дисперсии B: ({leftDB}, {rightDB})')


def main():
    Pij = np.random.rand(N, M)
    Pij = np.vectorize(lambda x: x / Pij.sum())(Pij)

    print('Теоретическая матрица распределения:')
    print(Pij)
    print()

    countMatrix = np.array([[0 for t in range(M)] for x in range(N)])
    for _ in range(ITER_COUNT):
        (i, j) = generateValue(Pij)
        countMatrix[i][j] += 1
    normalizedCountMatrix = np.vectorize(lambda x: x / countMatrix.sum())(countMatrix)

    print('Эмпирическая матрица распределения:')
    print(normalizedCountMatrix)
    print()

    theorMAi, theorMBj = getM(Pij, Ai, Bj)
    empirMAi, empirMBj = getM(normalizedCountMatrix, Ai, Bj)
    print(f'Теоретическое мат. ожидание A: {theorMAi}; Эмпирическое мат. ожидание A: {empirMAi}')
    print(f'Теоретическое мат. ожидание B: {theorMBj}; Эмпирическое мат. ожидание B: {empirMBj}\n')

    squareThearMAi, squareThearMBj = getM(Pij, [x ** 2 for x in Ai], [x ** 2 for x in Bj])
    theorDAi, theorDBj = squareThearMAi - theorMAi ** 2, squareThearMBj - theorMBj ** 2
    squareEmpirMAi, squareEmpirMBj = getM(normalizedCountMatrix, [x ** 2 for x in Ai], [x ** 2 for x in Bj])
    empirDAi, empirDBj = squareEmpirMAi - empirMAi ** 2, squareEmpirMBj - empirMBj ** 2
    print(f'Теоретическая дисперсия A: {theorDAi}; Эмпирическая дисперсия A: {empirDAi}')
    print(f'Теоретическая дисперсия B: {theorDBj}; Эмпирическая дисперсия B: {empirDBj}\n')

    showPlots(Pij, normalizedCountMatrix)
    showCorrelation(normalizedCountMatrix, empirMAi, empirMBj, empirDAi, empirDBj)
    chi2(Pij, normalizedCountMatrix)
    intervalM(empirMAi, empirDAi, empirMBj, empirDBj)
    intervalD(empirDAi, empirDBj)


if __name__ == '__main__':
    main()
