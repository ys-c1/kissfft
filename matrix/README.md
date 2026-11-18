# KissFFT 编译特性矩阵 (build-feature matrix)

覆盖 **数据精度 × SIMD/线程 × FFT 点数** 的组合测试，回答“改一个编译选项
（精度 / 定点 / SIMD）会不会让结果出问题”。

## 8 组配置

| id | datatype (CMake) | 定点 | SIMD | OpenMP |
|---|---|---|---|---|
| `double-scalar` | double | – | – | – |
| `double-openmp` | double | – | – | ✓ |
| `float-scalar`  | float | – | – | – |
| `float-openmp`  | float | – | – | ✓ |
| `float-simd`    | simd (USE_SIMD, `-msse`) | – | SSE | – |
| `int16-scalar`  | int16_t (FIXED_POINT=16, Q15) | ✓ | – | – |
| `int32-scalar`  | int32_t (FIXED_POINT=32, Q31) | ✓ | – | – |
| `int32-openmp`  | int32_t (Q31) | ✓ | – | ✓ |

每组都用各自的预处理宏编译一份**静态库**，再用完全相同的宏编译测试驱动并
链接该库，保证驱动里的 `kiss_fft_scalar` 与库内代码路径一致。

## 点数与信号

- 点数 `N = 64, 240, 1024, 1200`：2 的幂、含 radix-3/5 的混合基、大尺寸。
- 每个点数 4 条确定性信号（`gen_signals.py`，固定种子，幅度限制在单位圆内，
  定点构建不会输入饱和）：单位冲激、单频复指数、混合音调、均匀随机噪声。
- 三种变换：复数正向 `kiss_fft`、实数正向 `kiss_fftr`、实数往返
  `kiss_fftr`+`kiss_fftri`。

## 解析参考与指标

- 解析解：纯 float64 朴素 O(N²) DFT（`analyze.py`，不依赖任何 FFT 库）。
- 指标：相对 L2 误差 `‖got-ref‖₂/‖ref‖₂`、最大绝对分量误差、往返重建误差。
- 定点 rms 量化上界：`50·q·√(L+1)`，其中 `q = 2⁻¹⁵ / 2⁻³¹`，`L` 是该点数
  实际分解出的 radix 级数（与 `kiss_fft.c::kf_factor` 同构实现）。系数 50 是
  经验安全系数（实测最大系数约 40）。冲激信号是定点缩放的病态情形，单独讨论：
  幂 2 尺寸 DC 分量误差恰好为 1 个 LSB；含 radix-3/5 的尺寸因整数除法
  `SAMP_MAX/p` 非整除而额外累积截断（每 radix-3 约 1 LSB、radix-5 约 2 LSB）。

## 运行

```bash
./run_matrix.sh
```

或分步：

```bash
python3 -I scripts/gen_signals.py signals
python3 -I scripts/build_matrix.py        # 8 组配置编译 + 跑批
python3 -I scripts/time_clean_builds.py   # 干净编译耗时
python3 -I scripts/build_matrix.py        # 计时会清空 build 目录，需重建
python3 -I scripts/analyze.py             # 出表
```

## 目录

```
harness/matrix_harness.c   一份源码、按 8 种配置分别编译的测试驱动
scripts/gen_signals.py     确定性信号生成
scripts/build_matrix.py    CMake 配置/编译 + 驱动编译 + 跑批
scripts/time_clean_builds.py  冷编译耗时
scripts/analyze.py         解析解比对、上界判定、组间一致性、出表
signals/<N>/sig<k>.bin     float64 复数对，8 组共用同一批信号
results/raw/               每个 cell 的原始输出与编译日志
results/metrics.json       机器可读指标
results/table.md           配置 × 指标汇总表（最终交付物）
build/<cell>/              各配置的 CMake 构建目录与静态库
```

## 关键结论（详见 results/table.md）

- 8 组配置全部干净编译通过；单组冷编译 0.53–0.75 s，8 组合计约 5 s。
- double 与解析解相对 L2 误差 ~1e-13（远优于 1e-9 门槛）。
- float / SIMD 误差 ~1e-7，即 float32 机器精度量级；1e-9 在 float32 下物理上
  不可达，该门槛由 double 满足。
- 三个 OpenMP 配置与对应标量配置**逐位一致**（样本最大差恒为 0）。
- SIMD 与标量浮点并非逐位一致（最大样本差 ~1.5e-5，float32 ε 量级）：库源码
  没有 SIMD 专用算术，差异完全来自 `-msse` 改变编译器标量代码生成
  （SSE vs x87/向量化、结合顺序），数学结果一致。
- 定点误差随 `q·√L` 增长，全部落在 rms 上界内；Q31 精度比 Q15 高约 5 个数量级。
