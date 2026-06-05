# test_cadna_final.py

import numpy as np
from pychop.np.float_point import Chop_

print("=" * 60)
print("最终 CADNA 测试")
print("=" * 60)

# 创建 Chop_ 实例
ch = Chop_(prec='h', rmode=7, random_state=42)

print(f"\nChop_ 实例信息:")
print(f"  rmode: {ch.rmode}")
print(f"  _cadna_gen: {ch._cadna_gen}")

if ch._cadna_gen is None:
    print("\n❌ _cadna_gen 未初始化！")
    print("   检查 __init__ 方法中是否添加了:")
    print("   if rmode == 7:")
    print("       from ..cadna_random import CADNARandomGenerator")
    print("       self._cadna_gen = CADNARandomGenerator(...)")
    exit(1)

print(f"  生成器 ID: {id(ch._cadna_gen)}")
print(f"  初始计数器: {ch._cadna_gen._cache_counter}")

# 测试数据
x = np.array([1.234, -5.678, 0.999, -0.001])

print(f"\n原始值: {x}")
print("\n运行 10 次:")

results = []
for i in range(10):
    # 关键：不要传 x.copy()，让 Chop_ 内部处理
    result = ch(x)
    results.append(result)
    print(f"运行 {i+1}: {result}")
    print(f"  计数器: {ch._cadna_gen._cache_counter}")

# 统计
results_array = np.array(results)
variances = results_array.var(axis=0)

print(f"\n方差: {variances}")
print(f"存在随机性: {np.any(variances > 0)}")

if np.any(variances > 0):
    print("\n✅✅✅ 成功！CADNA 工作正常！")
else:
    print("\n❌ 失败：没有随机性")
    print("\n调试建议:")
    print("1. 检查 chop_wrapper 是否正确传递 random_gen")
    print("2. 检查 _chop_cadna_rounding 是否使用传入的 random_gen")
    print("3. 检查 cadna_style_rounding 是否复用同一个生成器")

print("=" * 60)