import numpy as np
import sys
import os

# 确保导入路径正确
sys.path.insert(0, '/Users/chenxinye/pychop')

print("=" * 60)
print("CADNA 深度调试")
print("=" * 60)

# ============================================================
# 步骤 1: 测试 CADNA 生成器本身
# ============================================================
print("\n步骤 1: 测试 CADNA 随机数生成器")
print("-" * 40)

from pychop.cadna_random import CADNARandomGenerator

gen = CADNARandomGenerator(seed=42, backend="numpy")

# 生成多批随机位
print("生成 5 批随机位（每批 4 个）:")
for i in range(5):
    bits = gen.random_bits((4,))
    print(f"批次 {i+1}: {bits}")

print("\n如果每批都相同 → 生成器有问题")
print("如果每批不同 → 生成器正常\n")

# ============================================================
# 步骤 2: 测试位翻转
# ============================================================
print("\n步骤 2: 测试位翻转函数")
print("-" * 40)

from pychop.cadna_random import numpy_bit_flip

x = np.array([1.5, -2.3, 3.7, -0.5])
random_bits = np.array([0, 0, 1, 1], dtype=np.uint8)

x_flipped = numpy_bit_flip(x.copy(), random_bits)

print(f"原始值:     {x}")
print(f"随机位:     {random_bits} (1=翻转, 0=保持)")
print(f"翻转后:     {x_flipped}")
print(f"期望值:     [ 1.5 -2.3 -3.7  0.5]")

if np.allclose(x_flipped, np.array([1.5, -2.3, -3.7, 0.5])):
    print("✅ 位翻转正确\n")
else:
    print("❌ 位翻转错误\n")

# ============================================================
# 步骤 3: 测试 cadna_style_rounding
# ============================================================
print("\n步骤 3: 测试 cadna_style_rounding")
print("-" * 40)

# 直接从模块导入
import pychop.np.float_point as fp_module

# 创建生成器
gen_test = CADNARandomGenerator(seed=42, backend="numpy")

x = np.array([1.234, -5.678, 0.999, -0.001])

print(f"原始值: {x}")
print("\n使用同一个生成器运行 5 次:")

for i in range(5):
    result = fp_module.cadna_style_rounding(
        x.copy(), 
        flip=0, 
        p=0.5, 
        t=24, 
        randfunc=None, 
        random_gen=gen_test
    )
    print(f"运行 {i+1}: {result}")
    print(f"  生成器计数器: {gen_test._cache_counter}")

# ============================================================
# 步骤 4: 测试 Chop 类
# ============================================================
print("\n步骤 4: 测试 Chop 类")
print("-" * 40)

import pychop
from pychop import Chop

pychop.backend('numpy')

ch = Chop(exp_bits=5, sig_bits=10, rmode=7, random_state=42)

print(f"Chop 实例创建成功")
print(f"rmode: {ch.rmode}")
print(f"_cadna_gen: {ch._cadna_gen}")
print(f"_cadna_gen is None: {ch._cadna_gen is None}")

if ch._cadna_gen is not None:
    print(f"生成器 ID: {id(ch._cadna_gen)}")
    print(f"初始计数器: {ch._cadna_gen._cache_counter}")

# ============================================================
# 步骤 5: 测试 Chop 调用
# ============================================================
print("\n步骤 5: 测试 Chop 实际调用")
print("-" * 40)

x = np.array([1.234, -5.678, 0.999, -0.001])

print(f"原始值: {x}")
print("\n运行 5 次:")

for i in range(5):
    x_copy = x.copy()
    print(f"\n运行 {i+1}:")
    print(f"  调用前 - 输入: {x_copy}")
    print(f"  调用前 - 生成器计数器: {ch._cadna_gen._cache_counter if ch._cadna_gen else 'None'}")
    
    result = ch(x_copy)
    
    print(f"  调用后 - 输出: {result}")
    print(f"  调用后 - 生成器计数器: {ch._cadna_gen._cache_counter if ch._cadna_gen else 'None'}")

# ============================================================
# 步骤 6: 检查函数调用链
# ============================================================
print("\n步骤 6: 检查函数调用链")
print("-" * 40)

print("检查 ch._chop 函数:")
print(f"  函数: {ch._chop}")
print(f"  函数名: {ch._chop.__name__}")

# 尝试直接调用 _chop
print("\n直接调用 ch._chop:")
x_direct = x.copy()
result_direct = ch._chop(
    x_direct, 
    t=ch.t, 
    emax=ch.emax, 
    subnormal=ch.subnormal, 
    flip=ch.flip, 
    explim=ch.explim, 
    p=ch.p, 
    randfunc=ch.randfunc,
    random_gen=ch._cadna_gen
)
print(f"  结果: {result_direct}")
print(f"  生成器计数器: {ch._cadna_gen._cache_counter if ch._cadna_gen else 'None'}")

# ============================================================
# 步骤 7: 添加追踪
# ============================================================
print("\n步骤 7: 追踪函数调用")
print("-" * 40)

# 临时修改函数添加打印
original_cadna_round = fp_module.cadna_style_rounding

def traced_cadna_round(x, flip=0, p=0.5, t=24, randfunc=None, random_gen=None):
    print(f"    [TRACE] cadna_style_rounding 被调用")
    print(f"    [TRACE]   输入 x: {x}")
    print(f"    [TRACE]   random_gen is None: {random_gen is None}")
    if random_gen is not None:
        print(f"    [TRACE]   生成器 ID: {id(random_gen)}")
        print(f"    [TRACE]   计数器(前): {random_gen._cache_counter}")
    
    result = original_cadna_round(x, flip, p, t, randfunc, random_gen)
    
    if random_gen is not None:
        print(f"    [TRACE]   计数器(后): {random_gen._cache_counter}")
    print(f"    [TRACE]   输出: {result}")
    
    return result

# 替换函数
fp_module.cadna_style_rounding = traced_cadna_round

print("运行带追踪的 Chop:")
ch_traced = Chop(exp_bits=5, sig_bits=10, rmode=7, random_state=99)
x_test = np.array([1.234, -5.678])
print(f"\n输入: {x_test}")
result_traced = ch_traced(x_test.copy())
print(f"输出: {result_traced}")

# 恢复原函数
fp_module.cadna_style_rounding = original_cadna_round

# ============================================================
# 最终诊断
# ============================================================
print("\n" + "=" * 60)
print("诊断总结")
print("=" * 60)

print("\n请检查以下项目:")
print("1. 步骤 1: 生成器是否每次产生不同的随机位?")
print("2. 步骤 2: 位翻转是否正确?")
print("3. 步骤 3: cadna_style_rounding 是否产生不同结果?")
print("4. 步骤 5: Chop 调用是否产生不同结果?")
print("5. 步骤 7: random_gen 是否被正确传递?")

print("\n运行完整测试...")
results = []
for i in range(10):
    r = ch(x.copy())
    results.append(r)
    print(f"测试 {i+1}: {r}")

results_array = np.array(results)
variances = results_array.var(axis=0)

print(f"\n方差: {variances}")
print(f"存在随机性: {np.any(variances > 0)}")

if np.any(variances > 0):
    print("\n✅✅✅ 问题已解决!")
else:
    print("\n❌ 问题仍存在，请查看上述追踪输出找出原因")

print("=" * 60)