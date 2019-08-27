h, w = list(map(int, input().split()))
matrix = []
for i in range(h):
    matrix.append(list(map(int, input().split())))

m = int(input())
conv = []
for i in range(m):
    conv.append(list(map(float, input().split())))


def mat(m1, m2):
    res = 0
    n, m = len(m1), len(m1[0])
    for i in range(n):
        for j in range(m):
            res += m1[i][j] * m2[i][j]
    return res


def getM(matrix, i_start,i_end , j_start, j_end):
    ma = [[-1 for y in range(m)] for x in range(m)]
    for x in range(m):
        for y in range(m):
            ma[x][y] = matrix[x+i_start][y+j_start]
    return ma


def getRes(matrix, conv):
    res = [[-1 for j in range(h - m + 1)] for i in range(w - m + 1)]
    for i in range(w - m + 1):
        for j in range(h - m + 1):
            m1 = getM(matrix, i, i + m, j, j + m)
            res[i][j] = int(mat(m1, conv))
    return res


res = getRes(matrix, conv)
print('\n'.join([' ' .join([str(i) for i in x]) for x in res]))