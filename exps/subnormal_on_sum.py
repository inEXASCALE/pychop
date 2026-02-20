import numpy as np
import matplotlib.pyplot as plt
from pychop import Chop
import pychop

pychop.backend('numpy')   
np.random.seed(42)

ch_with = Chop(exp_bits=5, sig_bits=10, rmode=1, subnormal=True)   # rmode=1: round-to-nearest
ch_without = Chop(exp_bits=5, sig_bits=10, rmode=1, subnormal=False)

N = 1000
r = 0.99                  
scale = 2.5e-6    

k = np.arange(N)
terms = scale * (r ** k)

exact = scale * (1 - r**N) / (1 - r)
print(f"精确和 (double) = {exact:.12e}, N={N}, r={r}")

def accum_sum(ch):
    s = np.float64(0.0)
    history = []
    for t in terms:
        s = ch(s + t)          
        history.append(float(s))
    return np.array(history)

print("运行 subnormal=True ...")
history_with = accum_sum(ch_with)
print("运行 subnormal=False (flush-to-zero) ...")
history_without = accum_sum(ch_without)

print("\n" + "="*75)
print("几何级数小项累加实验结果（exp=5, sig=10）")
print("="*75)
print(f"With subnormal=True     : 最终累加值 = {history_with[-1]:.12e} | 误差 = {abs(history_with[-1] - exact):.2e}")
print(f"Without subnormal=False : 最终累加值 = {history_without[-1]:.12e} | 误差 = {abs(history_without[-1] - exact):.2e}")
print(f"误差比值 (False/True) ≈ {abs(history_without[-1] - exact) / abs(history_with[-1] - exact):.0f}x")

# ================== 绘图（直接放论文）==================
plt.figure(figsize=(6, 5))
plt.semilogy(np.abs(history_with - exact), label='subnormal=True (gradual underflow)', c='black', linewidth=2.5)
plt.semilogy(np.abs(history_without - exact), label='subnormal=False (flush to zero)', c='tomato', linewidth=2.5, linestyle='--')
plt.xlabel('Number of additions', fontsize=12)
plt.ylabel('Absolute error from exact sum', fontsize=12)
plt.title('Effect of Subnormal Support on Gradual Underflow\n'
          'Sequential summation in fp16', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/pychop_subnormal_geometric_sum.jpg', dpi=300, bbox_inches='tight')
plt.show()
