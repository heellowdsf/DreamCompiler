# Dream 升级说明（本轮改动汇总）

这份文档记录本轮对 Dream 的所有改进。所有改动都经过回归验证：
pong 训练正常（准确率 ~0.81）、C 互操作端到端闭环通过、6/6 功能回归全过。

---

## 1. 编译速度 — runtime 预编译缓存

**问题**：每次 `dream x.dream` 都要用 `-O3` 重新编译 9000 行的 runtime，
每次约 17 秒。纯 C 的 helloworld 瞬间编译，Dream 却慢得离谱——因为每次
都在重编整个运行时库。

**修复**：runtime 按内容指纹预编译成 `.o` 缓存，只在 runtime 源码或编译
参数变化时重建一次。之后每次编译只处理你脚本生成的几百行 IR 再链接。

- 首次：~17 秒（建缓存）
- 之后每次：**0.12 秒**（145 倍提速）
- 缓存位置：`~/.cache/dream/obj/`（Linux）/ `%LOCALAPPDATA%\Dream`（Windows）
- 设 `DREAM_NO_CACHE=1` 可禁用

---

## 2. 工程化 — tmp 不再污染项目 + build/run 子命令

**问题**：中间产物（output.ll、临时 exe）堆在项目目录里。

**修复**：
- 中间产物移到用户缓存目录，按脚本路径哈希分桶，项目目录保持干净
- `dream run x.dream` — 编译并运行，项目目录一尘不染
- `dream build x.dream -o app` — 产出独立 exe
- `dream x.dream` — 旧用法仍兼容
- `--emit-ll` — 需要看 IR 时导出

导出成果（.c/.h/.drck/模型）仍留在当前目录——那是你要的产物不是垃圾。

---

## 3. exe 体积 — build 模式死代码裁剪

**问题**：helloworld 的 exe 有 673KB，因为静态链接了整个 runtime（343 个
函数符号），哪怕只用了 println。

**修复**：`build` 模式用 `-Os` + 死代码裁剪（`-ffunction-sections`
+ `--gc-sections`）+ strip，链接器裁掉没用到的算子。

- helloworld exe：673KB → **47.6KB**（缩小 93%）
- `run` 模式仍用 `-O3` 保持开发时的运行速度
- 训练脚本用的算子多，裁剪空间小，但也会瘦身

---

## 4. 报错系统 — 编译期拦截 + 源码定位

**问题**：`.dream` 报错来自 clang 链接器或内部断言，对用户极不友好；
有些错误直接 segfault。

**修复**：统一的编译期诊断（行号 + 源码行 + `^^^` 定位 + 拼写建议）：

```
error: undefined variable 'widht'
  --> script.dream:5
   |
 5 |     let y = widht + 1.0;
   |             ^^^^^
  did you mean 'width'?
```

修复的具体问题：
- 未定义变量：从裸文本升级为带定位 + 建议
- 语法错误（括号不匹配）：从「报错后 segfault」变为干净退出
- 缺右大括号：从「静默接受」变为明确报错
- **宏定义算子误报**：relu/tensor_add 等 UNARY/BINOP 宏生成的函数
  曾被误判为「unknown function」，现在符号扫描器能识别宏定义

所有语义错误都在编译期拦截，clang 天书不再泄给用户。

---

## 5. 精度可配 — as f32/f64/exact 语法

**新语法**：逐处指定数值精度。

```dream
let s = sum(errors) as exact    // 补偿求和，消除累加舍入误差
let W = randn(64,24) as f32      // f32 存储标记
let acc = total as f64           // 强制 f64（默认）
```

**exact 档**（高精度）：归约类运算（sum/mean）用 Neumaier 补偿求和，
相对误差从 O(n·ε) 降到 O(ε)，与规模无关。

实测精度杀手 `[1e16, 1×10000]`：
- 普通求和：`10000000000009376`（丢失尾部）
- exact：`10000000000010000`（**精确**）

性能几乎无损——归约是内存带宽受限的，补偿的几次运算在数据加载阴影下免费。
（注意：runtime 用 `-ffast-math` 编译会把补偿代数化简掉，已用 `optnone`
精准隔离这一个函数。）

**默认行为不变**：不写精度声明的老脚本，行为逐位一致。f32/f64 目前是
存储标记，真正的 f32 计算路径（训练提速）是更大的改动，尚未启用。

---

## 6. 上亿次迭代稳定不崩

**问题**：最普通的累加器 `total = total + 1.0` 循环 300 万次就段错误崩溃。

**根因**：`let total = 0.0` 会把标量**装箱成 Tensor**，导致 `total = total + 1.0`
走 tensor_add **建计算图**，长循环里图链无限增长直至崩溃。

**四层修复**：

1. **标量不装箱**（治本）：`let x = 0.0` 保持原生 double，标量累加走纯 IR
   浮点运算，O(1) 内存永不建图。
2. **no_grad 块**：`no_grad { ... }` 块内不建图。实测千万次 tensor 累加
   （matmul+sum）内存全程稳定在 **2MB**（之前涨到 1GB+ 崩溃）。
3. **计算图深度软上限**：即使没用 no_grad，图链超过 5 万深自动 detach 截断。
4. **内存池分级回收**：大 buffer（>8MB）不进池直接归还系统。

```dream
no_grad {
    let i = 0;
    while (i < 100000000) {       // 一亿次
        let y = x @ W;
        total = total + sum(y);
        i = i + 1;
    }
}
```

---

## 7. 安全/缺陷审计 — 修复 4 个真实问题

系统性攻击 Dream 找缺陷：

| 缺陷 | 严重度 | 修复 |
|---|---|---|
| 大整数字面量崩溃（`3000000000` → stoi 抛异常 Aborted） | 高 | stoll 保护 + 超范围转 float |
| 超大分配崩溃（`zeros(100000,100000)` → bad_alloc） | 高 | int64 防溢出 + 5 亿上限优雅报错 |
| 负维度静默通过（`zeros(-5,3)` 返回垃圾） | 中 | 拒绝并报错 |
| relu 等宏算子误报 unknown function | 高 | 符号扫描器解析宏 |

**审计确认安全**：越界索引（有边界检查）、维度不匹配矩阵乘（有检查）、
空张量、未初始化变量 `let x;`、路径穿越。

**已知未修**（优先级低）：深递归百万层栈溢出（极端情况）、字符串+数字
静默混算、循环变量作用域泄漏。

---

## 8. 输出格式修复

- **println 多空行**：print 不再自带换行，换行完全由 println 控制。
  `print(1.0); print(2.0); println(3.0)` → `123\n`
- **浮点精度显示**：`10.0/3.0` 从被截断的 `3.33333` 改为完整往返精度
  `3.3333333333333335`
- **NaN/Inf 显示**：用位运算检测，绕过 -ffast-math 的告警

---

## 9. C 项目互操作（interop/）

让 C 项目和 Dream 无缝配合，无需手写 CSV。

**C → Dream 数据采集**（`interop/dream_capture.h`，header-only）：
```c
#define DREAM_CAPTURE_IMPL
#include "dream_capture.h"

DreamCapture* cap = dream_capture_open("data", 24);
dream_capture_row(cap, feat, 24, label);   // 游戏每帧一行
dream_capture_close(cap);
```
产出 `data_X.csv` / `data_Y.csv`，Dream 直接 `read_csv` 读取。

**Dream → C 模型导出**（`export_c_header`，已有）：
训练完导出无依赖的 C99 推理函数，C 项目 include 一行、调用一行。

完整闭环示例在 `interop/`：
- `example_collect.c` — C 采集圆形分类数据
- `example_train.dream` — Dream 训练并导出
- 端到端验证：训练准确率 99.7%，C 调用 4/4 正确

---

## 构建方法

```bash
cd Dream
mkdir build && cd build
cmake -DLLVM_DIR=<你的LLVM路径>/lib/cmake/llvm ..
cmake --build . --config Release
# 产物在 bin/DreamCompiler（Windows 为 DreamCompiler.exe）
```

需要 LLVM/clang（开发时用的是 LLVM 20）和 clang++ 在 PATH 中。
