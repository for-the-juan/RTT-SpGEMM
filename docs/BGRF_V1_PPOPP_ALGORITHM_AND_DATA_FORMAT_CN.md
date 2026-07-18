# BGRF-v1：面向 FP64 Tensor Core SpGEMM 的有界粗—细粒度双轴重排

## PPoPP 风格算法与数据格式说明

> 文档版本：2026-07-18  
> 实现对象：当前工作区启用的 BGRF-v1，而非 <code>#if 0</code> 中的旧版 unified/UTBR  
> 主要实现入口：<code>src/dmma_reorder.h:2498–2782</code>  
> 当前 <code>src/dmma_reorder.h</code> SHA-256：<code>fa560cbf8c4271a96dc4b6e4d3bf58a51d102c4db7b919c8b320482e377660e9</code>

## 摘要

RTT-SpGEMM 将输入矩阵 A、B 和输出矩阵 C 分别映射到 8×4、4×8 和
8×8 的 tile，并通过 NVIDIA FP64 <code>m8n8k4</code> DMMA 执行数值计算。
标量非零若分散在大量 A tiles 中，会直接增加 A 的 tile 元数据，并可能放大
A/B tile 交集和后续 Tensor Core 工作。BGRF-v1（Balanced GPU Reorder and Fusion）
针对这一粒度失配，联合构造 A 的行置换与 K 维置换。

算法首先把 A 的结构解释为“row–K”二部图，通过有界的 degree-bucket
Cuthill–McKee 遍历和全序反转产生一个 RCM-like 全局候选。该候选只有在
精确占用 8×4 A-tile 数严格下降时才被整体提交。随后，算法依次在固定
32-row 和 16-K 窗口中执行一次 Tensor Core 感知的贪心组包；每个窗口仅在
精确 tile 数下降，或 tile 数持平且 span/fanout 代理严格下降时提交。由此可得
一个实现级硬不变量：最终 A-tile 数不大于 identity 布局。

最终置换不显式生成一份完整的重排标量 CSR。实现将置换融合进 A 的 tile-key
生成、动态 B 的行映射以及 C 的行恢复。A 和 B 均采用 32-bit occupancy mask
与混合 payload：默认 tile 内唯一结构位置达到 24/32 时保存 32 个 FP64
物理槽，否则按 mask-bit 次序紧凑保存。本文给出算法形式化、伪代码、性质、
复杂度、端到端数据流和逐字段数据格式。

## 1. 背景、目标与范围

### 1.1 Tensor Core 执行粒度

当前数值路径固定使用：

| 操作数 | 逻辑 tile | tile 内布局 | WMMA 角色 |
|---|---:|---|---|
| A | 8×4 | row-major | <code>matrix_a</code> |
| B | 4×8 | column-major | <code>matrix_b</code> |
| C | 8×8 | row-major 输出 | <code>accumulator</code> |

一个 A tile 和一个 B tile 均有 32 个物理槽。相同 K-tile 上的一对 A/B
tiles 产生一个 8×8 累加。稀疏 tile 与稠密 tile 在存储方式上不同，但进入
数值核函数前都会被展开为完整的 32 个 FP64 槽，并调用同一个
<code>wmma::mma_sync</code>。

实现依据：

- tile 常量：<code>src/dmma_tiles.h:18–22</code>；
- FP64 fragment 与统一 DMMA：<code>src/dmma_spgemm.h:1279–1391</code>。

### 1.2 为什么需要双轴重排

仅重排行会改变哪些标量行被装入同一个 8-row group，但不能改变每组所覆盖的
4-K panels；仅重排 K 维也有同样局限。BGRF 因而联合寻找：

\[
\pi_r:[0,m)\rightarrow[0,m),\qquad
\pi_k:[0,k)\rightarrow[0,k),
\]

使结构相似的行更可能进入同一 8-row group，使结构相似的 K indices 更可能
进入同一 4-K group。

该设计同时处理两个尺度：

1. 全局图尺度：建立跨远距离 row/K IDs 的连通局部性；
2. Tensor Core 尺度：把标量 IDs 对齐到 8×4 tile 边界。

单纯的全局图排序可能在标量带宽上较好，却切坏 8/4 的 tile 边界；单纯的局部
组包又无法跨越原始宏观顺序。粗—细两阶段分别解决这两个问题。

### 1.3 优化范围

BGRF-v1 的硬目标是 A-only。其设计目标可写为：

\[
\operatorname{reduce}_{\pi_r,\pi_k}\ T_A(\pi_r,\pi_k),
\]

其中 \(T_A\) 是精确占用的 8×4 A-tile 数；当前启发式不声称求得全局最优解。
算法不直接优化：

- B 的 4×8 tile 数；
- C candidate 或 exact-output tile 数；
- DMMA tile-pair 数；
- 动态 B 重建时间；
- C 导出和行恢复时间；
- 端到端执行时间。

因此，“最终 A tiles 不增加”是可证明性质；“B/C 工作量或运行时间一定下降”
不是当前算法可作出的数学保证。

## 2. 问题定义与符号

令

\[
A\in\mathbb{R}^{m\times k},\qquad
B\in\mathbb{R}^{k\times n},\qquad
C=AB.
\]

本文区分两个非零计数：

- \(z\)：CSR 中的原始 entry 数；重复坐标和显式零均计入；
- \(\Omega_A\)：A 的唯一结构坐标集合，重复 \((i,j)\) 只出现一次。

主要符号如下。

| 符号 | 含义 |
|---|---|
| \(m,k,n\) | A 的行数、共享 K 维、B 的列数 |
| \(z\) | A CSR entry 数 |
| \(\pi_r(i)\) | old row \(i\) 的 new row ID |
| \(\pi_k(j)\) | old K ID \(j\) 的 new K ID |
| \(M_t=8\) | A/C 的标量行 group 大小 |
| \(K_t=4\) | A/B 的共享维 group 大小 |
| \(N_t=8\) | B/C 的标量列 group 大小 |
| \(W_r=32\) | row fine window |
| \(W_k=16\) | K fine window |
| \(T_A(\pi_r,\pi_k)\) | 置换后精确占用的 A tiles |

### 2.1 置换代数

定义置换矩阵 \(P_r\) 和 \(P_k\)：

\[
(P_r)_{\pi_r(i),i}=1,\qquad
(P_k)_{\pi_k(j),j}=1.
\]

则重排后的输入为

\[
A'=P_r A P_k^\mathsf{T},\qquad B'=P_kB.
\]

于是

\[
C'=A'B'
   =P_r A P_k^\mathsf{T}P_kB
   =P_rAB
   =P_rC,
\]

最终只需恢复输出行：

\[
C=P_r^\mathsf{T}C'.
\]

这个等式解释了三个实现选择：

1. B 只重排其行，即共享 K 维；
2. B 的列保持原坐标，所以 C 的列无需恢复；
3. C 在 tile-to-CSR 边界使用 row new-to-old 映射恢复原行序。

### 2.2 精确 A-tile 目标

置换后的 A-tile 集合定义为

\[
\mathcal{T}_A(\pi_r,\pi_k)=
\left\{
\left(
\left\lfloor\frac{\pi_r(i)}{8}\right\rfloor,
\left\lfloor\frac{\pi_k(j)}{4}\right\rfloor
\right)
:(i,j)\in\Omega_A
\right\}.
\]

精确 tile 数为

\[
T_A(\pi_r,\pi_k)=
\left|\mathcal{T}_A(\pi_r,\pi_k)\right|.
\]

实现为每个 CSR entry 生成 tile key，随后执行 sort+unique。因而重复坐标不会
重复计 tile，显式零仍保留其结构位置。

### 2.3 次级 span/fanout 代理

对某个 fine window \(w\) 中的 Tensor groups \(g\)，令
\(\mathcal{O}_{w,g}\) 是该 group 触及的对轴 tile ID 集合。定义

\[
\operatorname{span}(\mathcal{O})=
\begin{cases}
0, & \mathcal{O}=\varnothing,\\
\max(\mathcal{O})-\min(\mathcal{O})+1, & \text{otherwise},
\end{cases}
\]

以及

\[
F_w=\sum_g \operatorname{span}(\mathcal{O}_{w,g}).
\]

row fine 阶段的对轴是 K tiles；K fine 阶段的对轴是 row tiles。该量是连续
包围区间长度之和，而不是真实边数、唯一邻居数或精确 B/C 成本。本文因此称其为
span/fanout proxy。

## 3. 算法总览

完整数据流可概括为：

~~~text
                         Offline / stable A
Host CSR A
   │
   ├─ build row–K bipartite graph
   │      └─ bounded bucketed CM traversal + global reversal
   │             └─ joint row/K coarse proposal
   │
   ├─ exact global 8×4 A-tile guard
   │
   ├─ 32-row profile, greedy packing, exact per-window guard
   │
   ├─ 16-K profile, greedy packing, exact per-window guard
   │
   └─ fuse πr/πk into A entry keys → 8×4 Hybrid Tile-CSR

                         Online / dynamic B
CSR B ── fuse πk into B row keys ── sort/reduce
      └────────────────────────────→ 4×8 Hybrid Tile-CSR + Tile-CSC
                                      │
                                      ▼
                                uniform m8n8k4 DMMA
                                      │
                                      ▼
                                 8×8 C′ tiles
                                      │
                           fuse πr⁻¹ into tile-to-CSR
                                      ▼
                              original-order CSR C
~~~

算法固定参数为：

| 参数 | 当前值 |
|---|---:|
| fine sweeps | 1 |
| row window / group | 32 / 8 |
| K window / group | 16 / 4 |
| coarse degree buckets | 8 |
| fingerprint | 2×64 bit |
| coarse frontier-round budget | 2048 |

所有正常输入执行同一个固定流程。候选拒绝是算法目标函数的一部分，不是根据
矩阵类型切换到另一种重排方法。<code>--no-reorder</code> 是显式实验基线，
不属于 BGRF-v1 正常路径。

## 4. 阶段 I：有界二部图全局粗排

### 4.1 二部图构造

把 A 的结构表示为

\[
G=(R\cup K,E),
\]

其中

- \(R=\{r_0,\ldots,r_{m-1}\}\) 对应 A 的行；
- \(K=\{\kappa_0,\ldots,\kappa_{k-1}\}\) 对应 A 的列/K indices；
- 每个 CSR entry \((i,j)\) 形成边 \((r_i,\kappa_j)\)。

内部统一顶点编号为：

~~~text
row vertex:   0 ... m-1
K vertex:     m ... m+k-1
~~~

row degree 由 CSR row span 得到，K degree 通过 GPU atomic count 得到。这里的
degree 按原始 CSR entries 计数，所以重复坐标会提高启发式 degree；但邻接遍历
只处理排序后每组重复邻居的第一个，精确 tile guard 也会通过 unique 消除重复。

### 4.2 邻接顺序与 seed 顺序

粗排 degree bucket 为

\[
b(d)=
\begin{cases}
0,&d\le 1,\\
\min(\lfloor\log_2d\rfloor,7),&d>1.
\end{cases}
\]

每个顶点的邻居按

\[
(b(d_{\text{neighbor}}),\ \text{neighbor-id})
\]

排序。实现分别生成 row-side CSR 和 K-side CSC 的 64-bit keys，并执行两次
GPU sort。

连通分量 seed 使用全序：

\[
(\text{nonzero first},\ \text{smaller exact degree},\ \text{smaller id}).
\]

因此更准确的描述是“degree-bucket 邻接展开 + exact-degree seed order”，
而不是简单的全局 degree sort。

### 4.3 确定性的 level-synchronous CM 遍历

对每个 frontier：

1. 每个 parent 枚举尚未访问的唯一邻居；
2. 若多个 parents 竞争同一邻居，使用
   <code>atomicMin(owner, parent_position)</code> 选择 frontier 中最早的 parent；
3. 统计每个 parent 拥有的邻居数量并做 prefix scan；
4. 按 parent position、bucket 和 neighbor ID 的顺序写出下一 frontier。

每启动一个新分量，就选择 seed order 中第一个未访问的非零度顶点。遍历的
frontier rounds 在全图范围累计，最多为 2048，而不是每个分量各有 2048。
实现统计的 <code>coarse_levels</code> 是已处理的 frontier rounds，并包含
某个分量中产生空 next-frontier 的最后一轮；它不是图直径。

若达到预算但仍有活跃顶点，算法不切换到另一种方法。未访问顶点在 CM traversal
buffer 中按 reverse seed order 追加；随后对整个 traversal order 反转，因此
最终 RCM-like 候选中的 residual 保持确定性的 seed order。零度 row/K IDs
分别追加到对应轴的活跃序列之后。

由于当前实现是“CM traversal 后整体反转”，论文中应使用
“bounded bucketed CM with an RCM-like reversed proposal”，不宜直接称为标准
或完整 RCM。

### 4.4 联合 row/K 候选

全序反转后，算法从同一联合序列分别投影出：

\[
\pi_r^{c},\qquad \pi_k^{c}.
\]

这是标量级 row/K joint proposal。源码中虽然保留若干 Tensor-block projection
helper，但当前 <code>build_dmma_reorder_plan</code> 没有调用它们；这些 helper
不能写成现行算法步骤。

### 4.5 全局精确提交

粗排 incumbent 固定为 identity。每个 CSR entry 同时生成：

\[
q_{\text{cur}}=
\left\lfloor\frac{\pi_r(i)}8\right\rfloor
\cdot\left\lceil\frac{k}{4}\right\rceil+
\left\lfloor\frac{\pi_k(j)}4\right\rfloor,
\]

\[
q_{\text{prop}}=
2^{63}\ \vert\
\left(
\left\lfloor\frac{\pi_r^c(i)}8\right\rfloor
\cdot\left\lceil\frac{k}{4}\right\rceil+
\left\lfloor\frac{\pi_k^c(j)}4\right\rfloor
\right).
\]

对 \(2z\) 个 64-bit keys 排序并去重后，最高位把 current 与 proposal 的精确
tile 集分开。提交条件为

\[
\operatorname{accept}_{c}
\iff
T_A(\pi_r^c,\pi_k^c)
<
T_A(I_m,I_k).
\]

相等也拒绝。若接受，row 与 K 两个映射同时交换为 proposal；不允许只接受其中
一个轴。

## 5. 阶段 II：Tensor Core 感知的局部细排

### 5.1 Axis profile

在另一个轴的当前置换下，为每个标量轴元素 \(x\) 构造：

\[
\operatorname{profile}(x)=
(id,d,b,o,p_{\min},p_{\max},s,F_0,F_1).
\]

字段含义如下。

| 字段 | 含义 |
|---|---|
| \(id\) | 原始轴 ID |
| \(d\) | 原始 CSR entry degree |
| \(b\) | soft degree bucket |
| \(o\) | 两个 fingerprint word 的总 popcount |
| \(p_{\min},p_{\max}\) | 触及的最小/最大对轴 panel |
| \(s\) | \(p_{\max}-p_{\min}+1\)，空轴为 0 |
| \(F_0,F_1\) | 128-bit hashed panel-set fingerprint |

soft bucket 为

\[
b_{\text{soft}}(d)=
\begin{cases}
-1,&d=0,\\
\left\lfloor
\dfrac{\lfloor\log_2d\rfloor}{2}
\right\rfloor,&d>0.
\end{cases}
\]

对 panel \(p\)，每个 fingerprint word 通过固定 <code>mix64</code> 哈希选择
一个 bit 并执行 OR。因此 \(o\) 只是 panel cardinality 的近似量，哈希冲突会
低估真实数量。

<code>DmmaAxisProfile</code> 还计算两个 MinHash 值，但当前 anchor comparator
和 pack score 均未读取它们。论文算法不应声称使用了 MinHash。

### 5.2 Row fine proposal

row profiles 先按 incumbent 中的当前物理位置排序，然后每个连续 32-row
window 独立组包。完整窗口形成四个 8-row Tensor groups；末尾不足 32 的窗口
按实际元素数形成若干完整或部分 groups。

每组首先选择 anchor，优先级为：

1. nonempty first；
2. larger soft bucket；
3. larger approximate panel occupancy；
4. smaller panel minimum；
5. smaller panel span；
6. larger exact entry degree；
7. smaller fingerprint，按 word 字典序；
8. smaller original ID。

选定 anchor 后，逐个加入其余最多 7 个成员。候选 \(x\) 相对 anchor \(a\)
和当前 group aggregate 的评分顺序为：

\[
\left(
|b_a-b_x|,
\operatorname{newBits}(x),
-\frac{\operatorname{overlap}(x)}
        {\operatorname{union}(x)},
-\operatorname{overlap}(x),
\Delta\operatorname{span}(x),
|d_a-d_x|,
id_x
\right).
\]

实现按该字典序选择最优候选，其中：

- bucket distance 和 degree difference 相对 anchor 计算；
- new bits、overlap 和 union 相对当前 group 的 OR fingerprint 计算；
- span growth 相对当前 group 的精确 min/max panel 计算；
- overlap ratio 用交叉相乘比较，避免浮点数。

所有移动严格限制在 incumbent 的当前 32-row window 内。

### 5.3 Row window 的精确提交

对每个 entry，row 阶段构造的精确 key 编码：

\[
(\text{state},\ \text{32-row window},\
 \text{8-row group},\ \text{K-tile}).
\]

sort+unique 后，每个 unique key 恰好对应一个占用 A tile。对每个 window
\(w\)，计算 current/proposal 的 \(T_w\) 和 \(F_w\)。提交条件为

\[
\operatorname{accept}_r(w)
\iff
T'_w<T_w
\ \lor\
(T'_w=T_w\land F'_w<F_w).
\]

若 tile 数下降，则即使 fanout proxy 上升也会接受；只有 tile 数持平时才要求
fanout 严格下降。拒绝只回退该窗口，其他已接受窗口不受影响。

### 5.4 K fine proposal 与提交

row 阶段完成后，算法使用最终 row permutation 重新构建 K profiles。随后在每个
16-K window 内，以相同的 anchor 与 pack score 形成最多四个 4-K groups。

K 阶段的精确 key 编码：

\[
(\text{state},\ \text{16-K window},\
 \text{4-K group},\ \text{row-tile}),
\]

并使用同一个 \((T_w,F_w)\) 字典序提交规则。

执行顺序固定为 row 后 K，且只执行一轮；两个阶段不是并行、独立或迭代至收敛。

## 6. 论文式伪代码

### Algorithm 1：BGRF-v1 主流程

~~~text
Algorithm 1 BGRF-v1(A in CSR)
Input:
    A: m × k CSR with z entries
Output:
    πr, πk: full old-to-new bijections
    πr⁻¹, πk⁻¹: full new-to-old bijections

1:  (πr, πk) ← (Identity(m), Identity(k))
2:  (πrᶜ, πkᶜ) ← BoundedBipartiteRCMLike(A, buckets=8,
                                         levelBudget=2048)
3:  if ExactATiles(A, πrᶜ, πkᶜ) < ExactATiles(A, πr, πk) then
4:      (πr, πk) ← (πrᶜ, πkᶜ)
5:  end if
6:
7:  Pr ← BuildRowProfiles(A, πk)
8:  \hat{πr} ← GreedyPackInCurrentWindows(Pr, πr,
                                        window=32, group=8)
9:  πr ← ExactWindowCommit(A, πr, \hat{πr}, πk,
                          window=32, group=8)
10:
11: Pk ← BuildKProfiles(A, πr)
12: \hat{πk} ← GreedyPackInCurrentWindows(Pk, πk,
                                       window=16, group=4)
13: πk ← ExactWindowCommit(A, πk, \hat{πk}, πr,
                         window=16, group=4)
14:
15: ar ← SafeActivePrefix(Pr.degree, πr)
16: ak ← SafeActivePrefix(Pk.degree, πk)
17: materialize inverse maps πr⁻¹ and πk⁻¹
18: return (πr, πk, πr⁻¹, πk⁻¹, ar, ak)
~~~

### Algorithm 2：有界二部图 RCM-like proposal

~~~text
Algorithm 2 BoundedBipartiteRCMLike(A, buckets, levelBudget)
1:  build row and K entry-degrees
2:  build row-side CSR and K-side CSC adjacency
3:  sort each adjacency by (neighbor degree bucket, neighbor id)
4:  seedOrder ← sort active vertices by (degree, id)
5:  visited ← false; traversal ← empty; levels ← 0
6:
7:  while unvisited active vertices remain and levels < levelBudget do
8:      s ← first unvisited vertex in seedOrder
9:      frontier ← [s]; append s to traversal
10:     while frontier ≠ empty and levels < levelBudget do
11:         owner[v] ← earliest frontier parent that reaches v
12:         next ← stable concatenation of each parent's owned neighbors
13:         append next to traversal; frontier ← next
14:         levels ← levels + 1
15:     end while
16: end while
17:
18: if active vertices remain then
19:     append them to traversal in reverse seed order
20: end if
21: traversal ← reverse(traversal)
22: project row vertices and K vertices into two scalar orders
23: append zero-degree IDs to their corresponding axis order
24: return both old-to-new bijections
~~~

### Algorithm 3：窗口内组包与精确提交

~~~text
Algorithm 3 PackAndCommit(A, profiles, incumbent, window W, group G)
1:  order profiles by incumbent physical position
2:  for each disjoint current window w in parallel do
3:      unused ← all axis IDs in w
4:      while unused ≠ empty do
5:          a ← best anchor in unused
6:          start a new group with a; remove a from unused
7:          while group size < G and unused ≠ empty do
8:              x ← lexicographically best PackScore(x | a, group)
9:              append x to group; remove x from unused
10:         end while
11:     end while
12: end for
13:
14: generate exact current/proposal tile keys for every CSR entry
15: sort and unique all keys
16: for each window w in parallel do
17:     compute exact tiles (Tw, T'w) and span proxies (Fw, F'w)
18:     if T'w < Tw or (T'w = Tw and F'w < Fw) then
19:         commit only window w
20:     end if
21: end for
22: return the partially committed full bijection
~~~

## 7. 正确性与可证明性质

### Lemma 1：乘法等价性

若 \(\pi_r,\pi_k\) 为双射，且 B 的行使用与 A 的 K 维相同的 \(\pi_k\)，则

\[
P_r^\mathsf{T}
\left[
(P_rAP_k^\mathsf{T})(P_kB)
\right]
=AB.
\]

证明直接来自 \(P_k^\mathsf{T}P_k=I\) 和
\(P_r^\mathsf{T}P_r=I\)。因此重排不会改变数学结果；只恢复 C 的行即可。

### Lemma 2：Fine windows 相互独立

row proposal 不跨 32-row window，K proposal 不跨 16-K window。因此任一 A
tile 在某个 fine 阶段始终属于同一个 axis window；一个窗口的提交不会改变另一
窗口的 tile membership 或 exact key。故可以逐窗口独立接受或拒绝。

### Theorem 1：A-tile 单调性

令 \(T_0\) 为 identity A-tile 数，\(\Delta_c,\Delta_r,\Delta_k\) 分别为
coarse、row fine 和 K fine 的精确 tile 减少量。则

\[
T_{\text{final}}
=T_0-\Delta_c-\Delta_r-\Delta_k
\le T_0.
\]

证明如下：

1. coarse joint proposal 仅在全局 tile 数严格下降时提交；
2. 每个 row window 只允许 tile 数下降或持平；
3. 每个 K window 使用相同规则；
4. 由 Lemma 2，同一阶段的窗口 tile 集构成不交分区；
5. 三阶段的减少量因此可以望远镜式累加。

该定理不涉及 payload 大小、B tiles、C candidates 或运行时间。

### Lemma 3：安全 active prefix

最终 row active prefix 定义为

\[
a_r=
\begin{cases}
0,&\forall i,\ d_i=0,\\
1+\max_{d_i>0}\pi_r(i),&\text{otherwise}.
\end{cases}
\]

K 轴同理。于是所有结构 entry 都满足

\[
\pi_r(i)<a_r,\qquad \pi_k(j)<a_k.
\]

因此 prefix 外部必为空，可以安全裁剪；prefix 内部仍可能包含空 row/K IDs，
所以 <code>active_rows</code> 不是非空行数量。

### Proposition 1：确定性

在相同输入和参数下，当前结构置换是确定性的：

- seed、anchor 和 pack score 最终均以 ID 打破平局；
- 多 parent 邻居竞争选择最早 frontier position；
- 重复邻居排序后只取一次；
- residual 使用确定性的 seed order；
- exact guard 只依赖整数 key 集。

测试还对同一 fixture 重复构建并比较完整 row/K permutation。

## 8. 计算、同步与空间复杂度

令 \(V=m+k\)，\(z\) 为 A CSR entry 数。当前实现的主要操作为：

| 阶段 | 主要 work | 显式临时空间 |
|---|---:|---:|
| degrees 与 key generation | \(O(z+V)\) | \(O(z+V)\) |
| row/K adjacency sort | 两次 \(O(z\log z)\) | \(O(z)\) |
| seed sort | \(O(V\log V)\) | \(O(V)\) |
| bounded traversal | \(O(z+V)\) 内的已访问图工作 | \(O(z+V)\) |
| global exact guard | sort+unique \(2z\) keys | \(16z\) bytes用于 keys |
| row profile sort/pack | \(O(m\log m+mW_r)\) | \(O(m)\) |
| row exact guard | sort+unique \(2z\) keys | 复用 exact keys |
| K profile sort/pack | \(O(k\log k+kW_k)\) | \(O(k)\) |
| K exact guard | sort+unique \(2z\) keys | 复用 exact keys |

由于 \(W_r=32,W_k=16\) 为常数，总体渐近 work 为

\[
O(z\log z+V\log V),
\]

显式额外空间为

\[
O(z+V).
\]

这里需要注意三个实现细节：

1. fine pack kernel 每个 window 由一个 CUDA thread 串行完成 \(O(W^2)\) 的小规模
   贪心选择，不同 windows 并行；
2. coarse 每个 frontier round 都读取 next-frontier size，产生 host-visible
   同步；全图累计 rounds 被限制为 2048；
3. <code>reorder_peak_workspace_bytes</code> 是源码按显式数组计算的估计值，
   不包含 Thrust sort 内部 scratch、CUDA allocator bookkeeping 或运行时碎片，
   不应表述为实测的绝对峰值显存。

## 9. 端到端数据格式

### 9.1 Matrix Market 与 Host CSR

程序入口读取 coordinate Matrix Market。文件坐标从 1-based 转为 0-based；
pattern entry 的值置为 1；symmetric/hermitian 输入会在 Host 端展开非对角
镜像项。loader 也识别 real、integer 和 complex 字段，但 complex 当前只保留
实部。

Host <code>SMatrix</code> 中与输入 CSR 相关的字段为：

| 字段 | dtype | 长度/语义 |
|---|---|---|
| <code>m,n,nnz</code> | <code>int</code> | shape 与 CSR entry 数 |
| <code>rowpointer</code> | <code>int*</code> | \(m+1\) |
| <code>columnindex</code> | <code>int*</code> | \(z\)，0-based |
| <code>value</code> | <code>double*</code> | \(z\)，FP64 |
| <code>isSymmetric</code> | <code>int</code> | 原始 banner 属性；CSR 已展开 |

有效 CSR 必须满足：

\[
row\_ptr[0]=0,\quad
row\_ptr[i]\le row\_ptr[i+1],\quad
row\_ptr[m]=z,
\]

以及

\[
0\le col\_idx[e]<k.
\]

不要求列索引预排序，也不要求坐标唯一。A/B tile 构造会对映射后的标量
position key 做 reduce-by-key，同一坐标的值相加。

当前 benchmark 可执行程序在 Matrix Market 载入后把 value 依次覆盖为
<code>entry_id % 10</code>。BGRF 本身只读取结构，不读取数值；论文若讨论
数值正确性或原始数据值，必须与这一 artifact 行为区分。

### 9.2 Device CSR

上传后的 <code>DmmaOwnedDeviceCsr</code> 为：

~~~text
rows, cols, nnz                       int32 scalar metadata
row_ptr[rows + 1]                     int32
col_idx[nnz]                          int32
values[nnz]                           FP64 double
~~~

GPU 预处理通过静态断言要求 <code>MAT_PTR_TYPE=int</code>，所以 CSR、
tile count 和 payload offset 受 32-bit 范围约束。

### 9.3 Coarse 图中间表示

令 \(V=m+k\)。核心图数组为：

| 数组 | dtype | 长度 | 含义 |
|---|---|---:|---|
| <code>row_degree</code> | int32 | \(m\) | row entry degree |
| <code>col_degree</code> | int32 | \(k\) | K entry degree |
| <code>graph_degree</code> | int32 | \(V\) | 合并后的顶点 degree |
| <code>col_ptr</code> | int32 | \(k+1\) | K-side CSC pointer |
| <code>csr_neighbors</code> | int32 | \(z\) | row→K local IDs |
| <code>csc_neighbors</code> | int32 | \(z\) | K→row IDs |
| <code>adjacency_keys</code> | uint64 | \(z\) | CSR/CSC sort 时复用 |
| <code>seed_order</code> | int32 | \(V\) | 全序 seed IDs |
| <code>frontier</code> | int32 | \(V\) | 当前 frontier |
| <code>next_frontier</code> | int32 | \(V\) | 下一 frontier |
| <code>traversal_order</code> | int32 | active vertices | CM order |
| <code>owner</code> | int32 | \(V\) | parent position / <code>INT_MAX</code> |
| <code>visited</code> | uint8 | \(V\) | 访问标志 |

row-side 邻接 key 为

\[
[((row\cdot8)+bucket(colDegree))\cdot(k+1)+col],
\]

K-side key 同构地交换 row/K。两种 key 都必须装入 <code>uint64_t</code>。

### 9.4 置换计划

<code>DmmaReorderPlan</code> 持有四个逻辑映射；每个映射均有 Host 和 Device
副本。

| 映射 | 长度 | 精确语义 | 主要使用者 |
|---|---:|---|---|
| row old-to-new | \(m\) | old row → new row | A tile-key 生成 |
| row new-to-old | \(m\) | new row → old row | C 行恢复 |
| inner old-to-new | \(k\) | old K → new K | A 列与 B 行映射 |
| inner new-to-old | \(k\) | new K → old K | 逆映射、诊断 |

所有元素均为 0-based int32；每一对映射满足

\[
new\_to\_old[old\_to\_new[x]]=x.
\]

这些数组始终是完整轴双射，即使 tile build 只物化安全 active prefix。
在 values-only B 更新模式中，A tiles 和初始 B source mapping 构建完成后，
四个 Device maps 被释放；Host maps 保留到 plan 销毁，其中 row new-to-old
用于每轮 C 恢复。

初始 Host+Device permutation payload 为

\[
16(m+k)\ \text{bytes},
\]

即每个轴两份 Host 和两份 Device int32 arrays；释放 Device maps 后剩余
\(8(m+k)\) Host bytes。

plan 的标量元数据按以下语义分组：

| 字段组 | 字段 | 单位/含义 |
|---|---|---|
| 输入 | <code>rows, cols, nnz</code> | A 的原始 shape 与 CSR entry 数 |
| 范围 | <code>active_rows, active_inner</code> | 覆盖所有非空轴的安全 prefix 长度 |
| 算法 | <code>kind, algorithm, unified</code> | 当前正常路径为 unified / <code>bgrf-v1</code> |
| 固定参数 | <code>sweeps, row_window, inner_window</code> | 当前为 1、32、16 |
| 移动量 | <code>moved_rows, moved_inner</code> | old ID 与 new ID 不同的轴元素数 |
| 位移量 | <code>row_displacement, inner_displacement</code> | \(\sum_x|\pi(x)-x|\)，标量 ID 槽 |
| fine 接受 | <code>accepted_row_windows, accepted_inner_windows</code> | 接受的窗口数 |
| fine tile | <code>row_tile_reduction, inner_tile_reduction</code> | 精确占用 8×4 A-tile 减少量 |
| fine proxy | <code>*_fanout_before/after</code> | 接受规则处理前后的对轴 tile-span 槽数之和 |
| coarse 遍历 | <code>coarse_components, coarse_levels, coarse_level_budget</code> | 预算内启动的分量、已处理 rounds、固定预算 |
| coarse 接受 | <code>coarse_candidate_accepted, coarse_tile_reduction</code> | joint proposal flag 与精确 tile 减少量 |
| 时间 | <code>coarse_ms, fine_ms</code> | 毫秒；实现阶段计时 |
| 空间 | <code>reorder_peak_workspace_bytes</code> | 显式数组估计 bytes |
| 最终 A | <code>num_tiles, active_row_tiles, active_k_tiles</code> | tile extent 与精确 tile 数 |
| 最终 payload | <code>sparse_tiles, dense_tiles, payload</code> | tile 数；payload 的单位是 FP64 elements |

当 2048-round 预算触发时，<code>coarse_components</code> 只统计截断前实际启动
遍历的分量，不是 residual 所属全图连通分量的精确总数。

### 9.5 Fine profile 与 exact-key IR

row profile 长度为 \(m\)，K profile 长度为 \(k\)。每个
<code>DmmaAxisProfile</code> 包含：

~~~text
id, degree, soft_bucket, panel_occupancy
panel_min, panel_max, panel_span
fingerprint[2]                         uint64
minhash[2]                             uint64, currently unused by decisions
~~~

coarse 和 fine exact guard 复用

~~~text
exact_keys[2 * z]                      uint64
~~~

proposal key 的 bit 63 置 1。fine 还使用：

~~~text
exact_counts[2 * max_windows]          int32
exact_span_min/max[2 * max_windows * 4]
exact_fanout[2 * max_windows]          int32
exact_stats[8]                         uint64
~~~

八个统计槽依次为：

~~~text
row accepted windows
row tile reduction
K accepted windows
K tile reduction
row fanout before / after
K fanout before / after
~~~

### 9.6 Hybrid Tile-CSR 通用容器

A 和 B 共用 <code>DmmaDeviceTiles</code> 逻辑视图：

| 字段 | dtype | 长度 |
|---|---|---:|
| <code>tile_row_ptr</code> | int32 | tile-row-count + 1 |
| <code>tile_col_idx</code> | int32 | num-tiles |
| <code>value_offsets</code> | int32 | num-tiles + 1 |
| <code>masks</code> | uint32 | num-tiles |
| <code>values</code> | FP64 | payload-size |
| <code>tile_col_ptr</code> | int32 | B 上为 tile-col-count + 1 |
| <code>tile_row_idx</code> | int32 | B 上为 num-tiles |
| <code>csc_tile_ids</code> | int32 | B 上为 num-tiles |

<code>tile_col_idx</code> 在每个 tile row 内递增。B 的 Tile-CSC 同样按 tile
column 组织，<code>csc_tile_ids[p]</code> 指回 Tile-CSR 中的 tile ID，使
numeric kernel 能从 CSC 位置找到相同 mask 与 payload。

若记 tile-row-count 为 \(R\)、tile 数为 \(T\)、payload 元素数为 \(L\)，
不含对象标量字段和 allocator overhead 的 A 存储为

\[
4(R+3T+2)+8L\ \text{bytes}.
\]

若 B 的 tile-col-count 为 \(N\)，其额外 CSC 为

\[
4(N+2T+1)\ \text{bytes}.
\]

### 9.7 A：8×4 row-major tiles

A 使用：

\[
R_A=\left\lceil\frac{a_r}{8}\right\rceil,\qquad
K_A=\left\lceil\frac{a_k}{4}\right\rceil.
\]

对原 entry \((i,j)\)，先映射

\[
i'=\pi_r(i),\qquad j'=\pi_k(j),
\]

再计算

\[
tileRow=\left\lfloor i'/8\right\rfloor,\qquad
tileCol=\left\lfloor j'/4\right\rfloor,
\]

\[
physical=(i'\bmod8)\cdot4+(j'\bmod4).
\]

64-bit entry key 为

\[
key=((tileRow\cdot K_A+tileCol)\cdot32+physical).
\]

置换直接融合进 key generation；实现不会先物化完整 \(A'\) 标量 CSR。A 只需
Tile-CSR，不构建 Tile-CSC。

### 9.8 B：4×8 column-major tiles

设 B 的 logical shape 为 \(k\times n\)。其 old row \(j\) 通过
\(\pi_k(j)\) 映射，列保持不变。物理布局为

\[
tileRow=\left\lfloor \pi_k(j)/4\right\rfloor,\qquad
tileCol=\left\lfloor c/8\right\rfloor,
\]

\[
physical=(c\bmod8)\cdot4+(\pi_k(j)\bmod4).
\]

因此 B mask 的每连续 4 bits 对应同一输出列的四个 K 槽，正好满足 WMMA
column-major fragment。

B 同时持有 Tile-CSR 和 Tile-CSC。numeric kernel 归并：

~~~text
A.tile_col_idx in one A tile row       // ordered K-tile IDs
B.tile_row_idx in one B tile column    // ordered K-tile IDs
~~~

两者相等时，通过 <code>csc_tile_ids</code> 取出 B 的 CSR tile ID，解码
A/B payload 并执行 DMMA。

### 9.9 Dense/bitmask 混合 payload

设 tile \(t\) 的 offset 为

\[
o_t=value\_offsets[t],\qquad
\ell_t=value\_offsets[t+1]-o_t.
\]

默认阈值 \(\theta=24\)：

- unique structural positions \(\ge\theta\)：dense；
- unique structural positions \(<\theta\)：bitmask packed。

格式标签不另设数组，而由 span \(\ell_t\) 隐式编码：

\[
\ell_t=32\iff \text{dense payload}.
\]

dense tile 保存全部 32 个物理值，未占用槽写 0；sparse tile 只保存 mask 中
置位位置的值，按 physical bit 从低到高排列。mask 在两种格式中都保留，始终
描述精确结构。

任意 physical slot \(q\) 的统一解码为

\[
decode(t,q)=
\begin{cases}
values[o_t+q],&\ell_t=32,\\
0,&\ell_t<32\land mask_t[q]=0,\\
values[o_t+\operatorname{popc}(mask_t\ \&\ (2^q-1))],
  &\text{otherwise}.
\end{cases}
\]

这里的 dense/sparse 判定依据结构位置数，而不是值是否为零。重复坐标先相加；
即使相加结果为数值零，该位置仍由 mask 作为结构位置保留。

### 9.10 动态 B 的 source-to-payload 映射

初次 B structure rebuild 使用以下主要工作数组，并在 tile arrays 外持久保留
source mapping：

~~~text
entry_keys[nnz_B]                      uint64
entry_values[nnz_B]                    FP64
source_ids[nnz_B]                      int32
head_flags[active_entries]             int32
unique_ids[active_entries]             int32
source_to_payload[nnz_B]               int32, persistent
~~~

<code>source_to_payload[e]</code> 是原 B CSR entry \(e\) 在 packed
<code>values</code> 中的目标 offset。若多个 source entries 是重复坐标，它们
指向同一 offset；无效或位于安全 active K prefix 外的 entry 使用 -1。

values-only 更新执行：

1. 清零整个 B payload；
2. 每个 source entry 按 <code>source_to_payload</code> scatter；
3. 有重复坐标时使用 <code>atomicAdd</code>，否则直接 store。

因此 values-only 路径不重做 permutation、sort/reduce、mask、Tile-CSR 或
Tile-CSC。结构变化时则必须完整 rebuild。

<code>DmmaDynamicB</code> 还记录 source shape、<code>source_nnz</code>、
已分配的 mapping capacity、active entry 数、是否存在重复坐标以及 valid
状态。values-only 模式在初次构建后释放上述 sort/reduce workspace，但保留
tile arrays 与 <code>source_to_payload</code>。

对 AAT，B 是 A 的逻辑转置，key generation 交换 source row/column，不物化
完整 \(A^\mathsf{T}\) CSR；对 AA，B source 复用 A 的 Device CSR；一般 AB
使用独立 B CSR。

### 9.11 C tile 输出与行恢复

GPU 数值阶段导出的 C′ 为 8×8 tile-CSR：

| 字段 | dtype | 长度/含义 |
|---|---|---|
| <code>tile_ptr</code> | int32 | C tile rows + 1 |
| <code>tile_columnidx</code> | int32 | C tile count |
| <code>tile_nnz</code> | int32 | C tile count + 1，value offsets |
| <code>tile_csr_Ptr</code> | uint8 | 每 tile 8 个局部 row offsets |
| <code>tile_csr_Col</code> | uint8 | 每个输出值的局部列 0…7 |
| <code>tile_csr_Value</code> | FP64 | nnzC |

非 identity 时，tile-to-CSR 直接把 new row 的计数和值写入
<code>row_new_to_old[new_row]</code> 对应的原 row segment。active prefix
之外若发现非空输出会立即失败，不能静默丢弃。C 的 global column index 不做
置换恢复。

最终 Host CSR 为：

~~~text
rowpointer[m + 1]                      int32
columnindex[nnzC]                      int32, original column space
value[nnzC]                            FP64
~~~

### 9.12 磁盘诊断格式

<code>--dump-reorder-prefix P</code> 产生：

| 文件 | 格式与语义 |
|---|---|
| <code>P_permutations.csv</code> | row/inner 的 0-based old→new 与冗余 inverse 校验 |
| <code>P_unified_stats.csv</code> | 两列 key,value；时间、接受数、tile reduction、fanout、workspace |
| <code>P_A_reordered.mtx</code> | Matrix Market pattern general，仅结构，1-based 坐标 |

permutation CSV header 为：

~~~csv
axis,old_id,new_id,new_to_old_at_new_id
~~~

第四列应等于第二列，用于验证 inverse。重排 Matrix Market 保留原 CSR entry
数和重复坐标，不保存值；写出循环按 old row 顺序执行，因此文件行不保证按
new row 排序。

<code>--dump-reorder-heatmap P</code> 产生：

~~~csv
P_heatmap.csv:
bin_row,bin_col,original_nnz,reordered_nnz

P_heatmap_meta.csv:
key,value
~~~

grid 恰有 <code>bins²</code> 行并包含零格；两侧计数总和都应等于 runtime
CSR entry 数。注意它统计 entries，而不是去重后的结构坐标。

外部对比方法的 order 文件与上述 CSV 不同：C++ loader 要求恰好 \(m\) 或
\(k\) 个 0-based decimal tokens，第 <code>new_id</code> 个 token 表示
<code>old_id</code>，即 new-to-old 语义；短文件、额外 token、重复或越界均拒绝。

## 10. 小型示例

考虑一个 16×8 的 A，其唯一结构坐标为

\[
\Omega_A=\{(0,0),(1,4),(8,1),(9,5)\},
\]

对应值依次记为 \(a,b,c,d\)。

identity 布局产生四个 A tiles：

\[
(0,0),\ (0,1),\ (1,0),\ (1,1).
\]

假设某个 32-row fine proposal 把非空行映射为：

\[
\pi_r(0)=0,\quad
\pi_r(8)=1,\quad
\pi_r(1)=2,\quad
\pi_r(9)=3,
\]

其余空行构成完整双射的尾部，且 \(\pi_k=I\)。重排后四个 entries 为：

| old coordinate | new coordinate | A tile | physical |
|---|---|---|---:|
| (0,0) | (0,0) | (0,0) | 0 |
| (8,1) | (1,1) | (0,0) | 5 |
| (1,4) | (2,4) | (0,1) | 8 |
| (9,5) | (3,5) | (0,1) | 13 |

精确 A-tile 数从 4 降为 2，因此该 window proposal 会被提交。若 active row
prefix 为 4，A 的 sparse Hybrid Tile-CSR 为：

~~~text
tile_row_ptr  = [0, 2]
tile_col_idx  = [0, 1]
value_offsets = [0, 2, 4]
masks         = [0x00000021, 0x00002100]
values        = [a, c, b, d]
~~~

第一个 mask 的 bits 0 和 5 置位，第二个 mask 的 bits 8 和 13 置位。
<code>value_offsets</code> span 均为 2，因此两个 tiles 都使用 bitmask packed
格式。这个例子同时展示了：

- 行重排如何减少跨 row-tile 的重复 K panels；
- mask 的 row-major physical 编号；
- packed values 按 mask bit 次序存放；
- active prefix 可以短于完整轴，但 permutation 本身仍是完整双射。

## 11. 实现映射与验证

### 11.1 权威代码位置

| 主题 | 当前实现 |
|---|---|
| BGRF 常量、plan 与 coarse/fine | <code>src/dmma_reorder.h:38–1855,2498–2782</code> |
| 禁用的旧 unified builder | <code>src/dmma_reorder.h:2171–2496</code> |
| Host hybrid tile oracle | <code>src/dmma_tiles.h:18–440</code> |
| Device CSR、A/B tile build、动态 B | <code>src/gpu_dmma_tiles.h:29–1382,1553–1735</code> |
| Hybrid view、mask decode、DMMA | <code>src/dmma_spgemm.h:41–74,1279–1391</code> |
| C 行恢复 | <code>src/tile2csr.h:331–522</code> |
| 默认调用链与 dump | <code>src/main.cu:193–356,459–765</code> |

### 11.2 已编码的测试性质

测试覆盖：

- 乱序 CSR、重复坐标、zero-sum duplicate 和显式零；
- 23 sparse / 24 dense 阈值边界；
- partial 8×4 与 4×8 tiles；
- 矩形 AAT、0×0 与 nonzero-shape empty；
- 完整双射、Host/Device maps 一致和安全 active prefix；
- 相同输入重复构建的 permutation 确定性；
- 触发 2048-round residual path；
- dynamic B values-only 与 full rebuild 的 payload 等价性；
- C shortened prefix、非 identity 恢复以及非法 permutation 拒绝。

本次核验在 A100 环境运行仓库现有测试二进制，结果为：

~~~text
dmma_dynamic_b_test:       PASS
gpu_dmma_tiles_test:       PASS
tile2csr_restore_test:     PASS
~~~

这些测试证明实现与 CPU tile oracle、映射约束和恢复路径一致；它们不替代对
大规模性能、数值误差或任意 B/C 工作量的实验评价。

## 12. 论文写作中的 claim 边界

| 可以由当前实现支持的表述 | 当前不能支持或容易误写的表述 |
|---|---|
| 有界 bucketed CM + reversed RCM-like proposal | “执行标准完整 RCM” |
| coarse joint row/K 候选由精确 A-tile 数守卫 | “coarse 直接优化 B/C” |
| fine fingerprint 用于低成本 proposal | “fingerprint 是精确 panel set” |
| tile 持平时使用 span/fanout proxy | “fanout 是真实邻居数或精确通信量” |
| final A tiles 不大于 identity | “B tiles、C candidates 或时间必然下降” |
| row 后 K，固定一次 sweep | “两个轴并行更新或迭代至收敛” |
| active prefix 覆盖所有非空轴 | “active 等于非空轴数量” |
| residual 在 2048 rounds 后确定性补齐 | “所有连通分量都完成了 BFS” |
| <code>coarse_components</code> 是预算内启动的分量数 | “预算触发时仍是全图精确分量数” |
| workspace 字段是显式数组估计 | “workspace 字段是实测绝对显存峰值” |
| profile 计算 MinHash 但当前决策未使用 | “BGRF-v1 使用 MinHash 评分” |
| dense threshold 只控制 payload 表示 | “24/32 阈值参与重排目标” |

还需明确以下局限：

1. A-only objective 无法对任意未来 B 保证 B/C 同时改善；
2. 128-bit fingerprint 有哈希冲突；
3. 固定 32/16 windows 限制跨窗口组合；
4. tile 减少时允许 span/fanout proxy 上升；
5. 2048-round 截断会牺牲长直径或大量分量图的完整全局遍历；
6. degree heuristic 计入重复 CSR entries，而 exact objective 对结构 key 去重；
7. row/K/tile/payload offsets 当前受 int32 上限约束；
8. benchmark 主程序当前覆盖输入 values，正式论文的数值验证口径需单独说明。

## 13. 可直接用于论文正文的总结段

BGRF-v1 separates global structural locality from Tensor Core boundary
alignment. It first performs a bounded, degree-bucketed traversal on the
row–K bipartite graph of A and reverses the resulting order to form one joint
RCM-like row/K proposal. An exact occupied-8×4-tile guard commits this proposal
only when it strictly improves the incumbent. BGRF-v1 then performs one
row-before-K fine sweep: it greedily packs structurally similar scalar IDs
inside fixed 32-row and 16-K windows into 8-row and 4-K Tensor groups,
respectively. Each window is committed only if its exact A-tile count
decreases, or if the tile count is unchanged and a fixed opposite-panel span
proxy decreases. Consequently, the final A layout never contains more
occupied tiles than the identity layout. The resulting permutations are fused
into A tile-key generation, online B-row mapping, and C-row restoration,
avoiding an explicitly materialized reordered scalar CSR.
