#
# nums = list(map(int,input().split()))
# n,m = nums[0],nums[1]
#
# nums1 = list(map(int,input().split()))
# nums2 = list(map(int,input().split()))
#
#
#
#
# def get_max_n(nums1,nums2):
#     if len(nums1)==1:
#         return str((nums1[0]+nums2[0])%m)
#     # res= 0
#     max_cur = -1
#     i_ind = j_ind = 0
#     for i in range(len(nums1)):
#         for j in range(len(nums1)):
#             if (nums1[i] + nums2[j])%m>=max_cur:
#                 max_cur = (nums1[i] + nums2[j])%m
#                 i_ind = nums1[i]
#                 j_ind = nums2[j]
#     nums1.remove(i_ind)
#     nums2.remove(j_ind)
#     return str(max_cur)+get_max_n(nums1,nums2)
#
# res = get_max_n(nums1,nums2)
# print(res)
#
#
# # def getRes(nums):
#

# matrix = []
# for i in range(9):
#     row = list(input())
#     matrix.append(row)
matrix = [["5","3",".",".","7",".",".",".","."],
          ["6",".",".","1","9","5",".",".","."],
          [".","9","8",".",".",".",".","6","."],
          ["8",".",".",".","6",".",".",".","3"],
          ["4",".",".","8",".","3",".",".","1"],
          ["7",".",".",".","2",".",".",".","6"],
          [".","6",".",".",".",".","2","8","."],
          [".",".",".","4","1","9",".",".","5"],
          [".",".",".",".","8",".",".","7","9"]]

import collections
def isValid(matrix):
    seen = []
    for i in range(9):
        for j in range(9):
            if matrix[i][j] != '.':
                val = (matrix[i][j])
                print('val',val)
                if (i, val) in seen or (val, j) in seen or  (i//3, j//3, val) in seen:
                    print('seen',seen)
                    print([(i, val), (val, j), (i//3, j//3, val)])
                    return False
                seen += [(i, val), (val, j), (i//3, j//3, val)]
                # print('seen',seen)
    # count = collections.Counter(matrix)
    # for key in count:
    #     if count[key]>1:
    #         print(key)
    return len(seen) == len(set(seen))


res = isValid(matrix)
if res:
    print('true')
else:
    print('false')

# def isValidSudoku(board) :
#     seen = []
#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             val = board[i][j]
#             print('val', val)
#             if i==3 and val==8:
#                 print()
#             if val != '.':
#                 seen += [(i, val), (val, j), (i // 3, j // 3, val)]
#     return len(seen) == len(set(seen))
#
# res = isValidSudoku(matrix)
# if res:
#     print('true')
# else:
#     print('false')


# 83..7....
# 8..195...
# .98....6.
# 8...6...3
# 4..8.3..1
# 7...2...6
# .6....28.
# ...419..5
# ....8..79

# .........
# .........
# .........
# .........
# .........
# .........
# .........
