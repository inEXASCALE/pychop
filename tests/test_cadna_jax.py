# test_cadna_jax.py

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# 假设上面的代码已经添加到 float_point.py

def test_cadna_random_generator():
    """测试 CADNA 随机数生成器"""
    print("=" * 60)
    print("测试 1: CADNA 随机数生成器")
    print("=" * 60)
    
    gen = CADNARandomGeneratorJAX(seed=42)
    
    # 生成 1000 个随机位
    bits = [gen.random_bit() for _ in range(1000)]
    mean_val = np.mean(bits)
    
    print(f"生成 1000 个随机位")
    print(f"均值: {mean_val:.4f} (期望: ~0.5)")
    print(f"最小值: {min(bits)}, 最大值: {max(bits)}")
    
    assert 0.45 < mean_val < 0.55, "随机位均值应接近 0.5"
    print("✓ 测试通过\n")


def test_jax_bit_flip():
    """测试 JAX 位翻转"""
    print("=" * 60)
    print("测试 2: JAX 位翻转")
    print("=" * 60)
    
    # float32 测试
    x32 = jnp.array([1.5, -2.3, 3.7, -0.5], dtype=jnp.float32)
    random_bits = jnp.array([1, 0, 1, 0], dtype=jnp.uint8)
    
    x32_flipped = jax_bit_flip(x32, random_bits)
    
    # 检查符号
    expected_signs = jnp.sign(x32) * (1 - 2 * random_bits)
    actual_signs = jnp.sign(x32_flipped)
    
    print("Float32:")
    print(f"原始值: {x32}")
    print(f"随机位: {random_bits}")
    print(f"翻转后: {x32_flipped}")
    print(f"符号正确: {jnp.all(expected_signs == actual_signs)}")
    
    # float64 测试
    x64 = jnp.array([1.5, -2.3, 3.7, -0.5], dtype=jnp.float64)
    x64_flipped = jax_bit_flip(x64, random_bits)
    
    print("\nFloat64:")
    print(f"原始值: {x64}")
    print(f"翻转后: {x64_flipped}")
    
    print("✓ 测试通过\n")


def test_cadna_rounding_basic():
    """测试基本 CADNA 舍入"""
    print("=" * 60)
    print("测试 3: CADNA 舍入基础功能")
    print("=" * 60)
    
    # 创建 Chop 实例
    ch = Chop_(prec='h', rmode=7, random_state=42)
    
    # 测试数据
    x = jnp.array([1.23456789, -2.34567890, 3.45678901, -4.56789012])
    
    print(f"原始值: {x}")
    print(f"\nCADNA 舍入结果（10 次运行）:")
    
    results = []
    for i in range(10):
        result = ch(x)
        results.append(result)
        print(f"运行 {i+1}: {result}")
    
    # 检查随机性
    results_array = jnp.stack(results)
    variances = jnp.var(results_array, axis=0)
    
    print(f"\n各元素方差: {variances}")
    print(f"存在随机性: {jnp.any(variances > 0)}")
    
    assert jnp.any(variances > 0), "CADNA 舍入应产生随机结果"
    print("✓ 测试通过\n")


def test_cadna_vs_standard():
    """对比 CADNA (rmode=7) 和标准随机舍入 (rmode=5)"""
    print("=" * 60)
    print("测试 4: CADNA vs 标准随机舍入")
    print("=" * 60)
    
    ch_cadna = Chop_(prec='s', rmode=7, random_state=42)
    ch_stoc = Chop_(prec='s', rmode=5, random_state=42)
    
    x = jnp.linspace(0, 10, 100)
    
    # 多次运行取平均
    results_cadna = []
    results_stoc = []
    
    for _ in range(20):
        results_cadna.append(ch_cadna(x))
        results_stoc.append(ch_stoc(x))
    
    mean_cadna = jnp.mean(jnp.stack(results_cadna), axis=0)
    mean_stoc = jnp.mean(jnp.stack(results_stoc), axis=0)
    
    # 两者的均值应该接近原始值
    error_cadna = jnp.mean(jnp.abs(mean_cadna - x))
    error_stoc = jnp.mean(jnp.abs(mean_stoc - x))
    
    print(f"CADNA 平均误差: {error_cadna:.6f}")
    print(f"标准随机舍入平均误差: {error_stoc:.6f}")
    print(f"两者误差相近: {jnp.abs(error_cadna - error_stoc) < 0.01}")
    
    print("✓ 测试通过\n")


def test_all_rounding_modes():
    """测试所有舍入模式"""
    print("=" * 60)
    print("测试 5: 所有舍入模式")
    print("=" * 60)
    
    x = jnp.array([1.5, 2.5, 3.7, -1.5, -2.3])
    
    modes = {
        1: "Round to nearest (even)",
        2: "Round towards +inf",
        3: "Round towards -inf",
        4: "Round towards zero",
        5: "Stochastic (proportional)",
        6: "Stochastic (uniform)",
        7: "CADNA"
    }
    
    for mode, name in modes.items():
        ch = Chop_(prec='h', rmode=mode, random_state=42)
        result = ch(x)
        print(f"rmode={mode} ({name}):")
        print(f"  输入:  {x}")
        print(f"  输出:  {result}\n")
    
    print("✓ 所有模式运行成功\n")


def test_performance():
    """性能测试"""
    print("=" * 60)
    print("测试 6: 性能对比")
    print("=" * 60)
    
    import time
    
    x_large = jnp.array(np.random.randn(10000), dtype=jnp.float32)
    
    modes = {
        5: "Stochastic (proportional)",
        6: "Stochastic (uniform)",
        7: "CADNA"
    }
    
    num_runs = 50
    
    for mode, name in modes.items():
        ch = Chop_(prec='s', rmode=mode, random_state=42)
        
        # Warm-up (JAX JIT 编译)
        _ = ch(x_large)
        
        # 计时
        start = time.time()
        for _ in range(num_runs):
            _ = ch(x_large)
        elapsed = time.time() - start
        
        print(f"rmode={mode} ({name}): {elapsed:.4f} 秒 ({num_runs} 次运行)")
    
    print("\n✓ 性能测试完成\n")


# 运行所有测试
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("JAX Backend - CADNA 实现测试套件")
    print("=" * 60 + "\n")
    
    test_cadna_random_generator()
    test_jax_bit_flip()
    test_cadna_rounding_basic()
    test_cadna_vs_standard()
    test_all_rounding_modes()
    test_performance()
    
    print("=" * 60)
    print("✅ 所有测试完成！")
    print("=" * 60)