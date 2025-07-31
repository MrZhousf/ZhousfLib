# -*- coding: utf-8 -*-
# @Author  : zhousf-a
# @Function:
import functools

""""
动态规划-斐波拉契数列
"""


def fib_force(n):
    # 暴力递归
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    return fib_force(n - 1) + fib_force(n - 2)


def fib_mem(n):
    global mem  # 备忘录算法
    # 自顶而下算法
    if n not in mem:
        mem[n] = fib_mem(n - 1) + fib_mem(n - 2)
    return mem[n]


@functools.lru_cache(None)
def fib_back(n):
    # 缓存备忘录算法 复杂度O(n)
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    return fib_back(n - 1) + fib_back(n - 2)


def fib_dp(n):
    # 动态规划
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    # db table
    dp = [0 for _ in range(n + 1)]
    dp[1] = dp[2] = 1
    # 自底而上算法
    for i in range(3, n + 1):
        # 状态转移方程
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]


if __name__ == "__main__":
    mem = {0: 0, 1: 1, 2: 2}
    print(fib_mem(6))
