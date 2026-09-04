# MHS-shaped spike: dose-aware capture + digitisation QC

时间盒 spike（〈一〉），**未 commit，未固化**。目的：在敲门之前手里有东西给人看。

## 证据分层（〈二〉）
| 层级 | 状态 |
|---|---|
| `tested` | ✅ 23 项测试通过；M1–M13 变异中 12 组变红（M11 未变红 → 该守卫已删除，见下） |
| VLM 真实运行 | ✅ 仅 gemini-3.7-flash 后端，合成帧，n=2；Claude 后端**从未执行** |
| 真实仪器 | ❌ 零。两个后端都是替身 |
| 真实对象 | ❌ 零。夹具是合成页面，缺陷是注入的 |
| 保护方判断 | ❌ 零。剂量预算数字是我编的，须由保护师给 |
| `committed` / `reviewed` / `deployed` | ❌ |

**不能说的话**：「QC 工具可用」「剂量规划有效」。目前只能说：在注入已知缺陷的合成批次上，八项检查各自命中且不误伤干净页。

## 布局
```
mhs_shim/          MHS 形状的抽象层（公告推断，无公开 spec）
  manifest.py      对象安全清单：lux·h 预算 + 台账 + 拒绝，由驱动强制而非 prompt
  device.py        read/write 原语 + 自然语言 tag 生成设备参考文档
  backends/
    folder_scanner.py  虚拟扫描仪（路 C）：采集文件夹当设备
    rti_sim.py         模拟 RTI 灯罩（路 B）：每次曝光扣预算，超支即拒
qc/
  checks.py        路 C：八项单帧检查 + 一项批次相对检查（target_region）
  agent.py         路 C 的循环：检查 → 分诊 → VLM 复核 → 重拍清单 → 报告
  vlm.py           VLM 分诊层：协议、Claude 后端、stub、发送选择策略
  vlm_gemini.py    第二后端，走本机 gemini-ask（本机唯一有凭证的通道）
  relief.py        路 B：光度立体重建 + 盲判据（条件数/稳定性/对比度）+ oracle 评分
  capture_planner.py 路 B 的核心：选下一个光位 + 盲停机 + 预算拒绝
  rti_experiment.py  三臂对照：全罩 / 顺序+停机 / 自适应+停机
  rti_figure.py      给保护师看的一页（PNG + PDF），把不利于自己的读法印在图上
fixtures/make_rti_stack.py  按几何合成光栈：刻痕高度场 + 朗伯着色 + 已知光向 + 真值掩膜
fixtures/          合成批次生成器，真值独立于检查
cli.py             CLI 面（MHS 三个控制面之一）
mcp_server.py      MCP 面（同上）
tests/             判别性测试
```

## 三层分工（这是整个设计的核心）
| 层 | 管什么 | 为什么归它 |
|---|---|---|
| 单帧规则 | 分辨率、格式、位深、清晰度、曝光、倾斜、出血、重复、缺号 | 可测量、确定、便宜，每帧都跑 |
| 批次相对规则 | 色标/比例尺缺失 | **单帧看不出来**。缺色标的那一帧本身没有任何毛病，只有跟同批比才暴露。任何逐帧判官（人或模型）结构上都做不了 |
| VLM 视觉 | 手/手套入镜、异物、装反、内容被遮、非书页、对象受力 | 需要看图，无法测量 |

**铁律：VLM 只能加问题，不能消问题。** 规则失败是测量结果，模型意见不能推翻测量。
`escalate()` 是单向的，变异测试 M5 专门守这条。

## 跑
```
python3 cli.py fixtures runs/demo_batch
python3 cli.py describe runs/demo_batch     # agent 动手前先读的设备参考
python3 cli.py qc runs/demo_batch --dpi 400

python3 cli.py fixtures runs/vlm_batch --kind vlm          # 规则看不见的缺陷
python3 cli.py qc runs/vlm_batch --vlm gemini --vlm-mode all
python3 cli.py qc runs/vlm_batch --vlm anthropic --vlm-mode flagged   # 需要 Anthropic 凭证

python3 cli.py rti-stack runs/rti_stack                  # 合成光栈
python3 cli.py rti runs/rti_stack                       # 三臂对照
python3 cli.py rti-figure runs/rti_stack                # 一页图，PNG + PDF

python3 -m pytest tests/ -q
```

### 成本控制（三处）
1. **只发选中的帧**：`flagged` 只解释规则已判的、`sample:N` 抽查、`all` 全审。
2. **发之前缩图**：最长边 1024、JPEG q80。手入镜或装反在 1024 完全够看，发 400 ppi 母片是白烧钱。
3. **回复被 schema 约束**，输出 token 近似恒定。
按 1024×768 估约 1,050 input token/帧。用 `claude-opus-5`（$5/1M in）约 $0.006/帧：
500 页全审 ≈ $3，`flagged` 模式通常低一个量级。

## 实测结果 A · 规则层（合成批次 12 文件，13 页中第 10 页故意缺失）
`accept 3 / review 1 / reprocess 2 / reshoot 6`，缺号 `[10]`，八类注入缺陷全部按预期命中，
干净页零误报。

## 实测结果 B · VLM 层（真实模型，非 stub）
后端 `gemini-ask`（gemini-3.7-flash），六帧全审，**独立跑两次**：

| 帧 | 归谁管 | 规则层 | VLM 看到 | 最终动作 |
|---|---|---|---|---|
| p001 / p002 | 干净 | accept | 无 | accept ✓ |
| p003 手套按页 | VLM 专属 | accept | run1 `foreign_object`+`obscured_content` / run2 `hand_or_glove`+`obscured_content` | **reshoot** ✓ |
| p004 装反 | 规则可测 | reshoot（target_region=0.043） | `wrong_orientation` | reshoot ✓ |
| p005 缺色标 | 规则可测 | reshoot（target_region=0.094） | 无（已从词表移除） | reshoot ✓ |
| p006 笔遗留 | VLM 专属 | accept | `foreign_object`+`obscured_content` | **reshoot** ✓ |

- **动作层面两次都 6/6 正确，干净帧零误报。**
- **标签不稳定，动作稳定**：p003 两次分别被叫成 foreign_object 和 hand_or_glove。
  所以标签只能当线索给人看，**不能拿去统计缺陷类型分布**。
- 实测 token：6 帧 8,958 in / 224 out，即 **1,493 in + 37 out 每帧**。
  换算 `claude-opus-5`（$5/$25 每 M）约 **$0.0084/帧**；500 页全审 ≈ $4.2。

## 三个只有真跑才会发现的问题
1. **色标缺失根本不该交给 VLM。** 第一版把 `missing_colour_target` 放进 VLM 词表，
   实测漏检。根因不是模型弱：**单帧判官看不到批次**，缺色标的那一帧自身毫无异常。
   改成批次相对的规则检查 `target_region` 后稳定命中。
   顺带发现它也抓装反的页（色标位置随之改变）——**规则能测的就不要花钱问模型**。
2. **批次相对检查必须会弃权。** 第一版在正常文本页上误报，因为它假设该区域跨帧恒定。
   加 `min_consistency` 守卫：批次里该区域本就不恒定时整项弃权。变异测试 M6 守这条。
3. **夹具画得不像，会把模型的对判成错。** 我画的"手套"两次分别被读成异物和手套，
   画的"笔"被读成箭头。**动作都对**。用标签算准确率会得出错误结论。

## 实测结果 C · 路 B 自适应采集（本轮新增）
不再只有安全侧。规划循环已存在：按 E-最优选下一个光位，重建不再变化即停，
全程不看真值（有 AST 结构测试守着）。

单个光栈（48 光位、320px、朗伯渲染刻痕）：

| 臂 | 张数 | 剂量占全罩 | d′ 保留 | 停因 |
|---|---|---|---|---|
| 全罩（标准做法） | 48 | 100% | 100% | 固定全序列 |
| 顺序 + 停机 | 9 | 19% | 96% | 重建已稳定 |
| 自适应 + 停机 | 7 | 15% | 97% | 重建已稳定 |

六个光栈（三档噪声 × 两个种子）：

| | 张数中位（范围） | 剂量占全罩 | d′ 保留 |
|---|---|---|---|
| 顺序 + 停机 | 12（9–15） | 25% | 97% |
| 自适应 + 停机 | 7（7–11） | 15% | 96% |

**自适应在 6/6 上张数更少**，d′ 相差 1 个百分点以内。
停机阈值从 0.5° 扫到 3.0°，结论一次都没翻转。

**诚实的读法：干活的主要是「停机」，不是「选点」。** 停机把剂量从 100% 压到 25%，
选点再从 25% 压到 15%。把这条线卖成「智能选光位」是夸大，它真正的卖点是「知道什么时候够了」。

预算耗尽与达标是两个不同的停因，代码里分开、报告里分开，有测试守着。

## 三个变异测试逼出来的修正
1. **色标缺失不该交给 VLM**（见上）。
2. **批次相对检查必须会弃权**（见上）。
3. **条件数守卫是装饰，已删除。** M11 变异（去掉守卫）不变红，说明零测试覆盖。
   随后三次尝试构造它必须生效的场景**全部失败**：近乎平坦的对象、单圈灯罩、常规光栈。
   在 `min_lights=4` 加连续两轮稳定的前提下它冗余。改为守住真正的不变量
   「不得在 min_lights 之前停机」，该不变量的变异能变红。`cond` 仍记录在每步里，
   因为值得看，但不值得据以分支。

## 已知局限
- **光栈是合成的**：朗伯 + 已知几何，物理上站得住，但不是真实石刻、不是真实灯罩。
  真实表面有次表面散射、镜面高光、各向异性，本实验一概没有。
- 剂量数字（2000 lux / 0.5 s / 预算）全部是我编的，须由保护师给。
- Claude 后端（`--vlm anthropic`）**按官方文档形状写好但从未执行过**：本机无 Anthropic 凭证
  （`ANTHROPIC_API_KEY` 未设、Keychain 无条目、`ant` 未安装）。有凭证后一条命令即可验证。
- 一次真实运行在后台模式下挂死，前台正常。加了 `stdin=subprocess.DEVNULL` 加固，
  但**构造的探针没能复现该挂起，成因未确认**——不要把这条当已解决。
- 每次曝光的剂量按「lux × 秒」算，真实灯罩需要实测照度。
- `estimate_skew` 是投影方差搜索，对无文字页面（图版）无效。
- RTI 后端只验证了安全路径，回放公开光栈是下一轮。

## 过程中真实抓到的一个 bug
第一版 `check_skew` / `check_margins` 用固定阈值 128 分墨与纸。欠曝页面整张纸都低于 128，
于是背景被当成内容，skew 与 margins 双双误报。测试抓到后改成随曝光自适应的 `ink_mask`：
以纸的中位数为背景、1 百分位为墨，阈值取两者之间。变异测试 M3 把它退回硬阈值，测试立刻变红。
