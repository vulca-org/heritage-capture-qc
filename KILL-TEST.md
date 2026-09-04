# MHS（Model Hardware Standard）能否落到人文领域 — kill test

日期：2026-09-02　性质：只读投资门检查（〈三〉）。未启动任何实现。

## MHS 是什么（一手：anthropic.com 公告）
- 标准化驱动层：设备以 read/write 原语暴露，自然语言 tag 生成设备参考文档，agent 经 MCP / CLI / 代码 API 编排。
- 状态：research preview，**waitlist**（modelhardwarestandard.com），面向「scientific research labs and advanced manufacturers」；**无公开 spec/SDK**；开源时间「ahead of making the standard open source」无日期。Raspberry Pi 与 HF LeRobot 在接入。
- 五个案例全在生物/显微/量子：Genentech 蛋白 assay、UW 远程监控+qPCR 择时停机、CMU 剂量曲线、QuEra 激光锁定、Janelia 七家厂商软件统一进一个面板。
- 自报局限：物理直觉缺失（气泡事故需人指导）、无编程接口的设备不支持、长时运行算力成本。

## 人文领域里「有可编程接口的硬件」在哪
只在**物质文化**一侧成立（纯文本/阐释类人文无硬件）：
1. 遗产科学/保护实验室：XRF/MA-XRF、FTIR、Raman、多光谱/高光谱、显微镜 — 与 Janelia「七家厂商互不兼容」同构。
2. 数字化工作室：书籍扫描仪、翻拍台、RTI 灯罩、摄影测量转台。
3. 音像迁移：磁带机（RS-232/RS-422）。
4. 博物馆环境监测。
5. 考古田野：无人机 GPR/磁力。
6. 同步辐射（Herculaneum 类）。

## 判断一：已被覆盖
| 落点 | 最近先例 | 结论 |
|---|---|---|
| agent↔仪器协议层 | MHS 本身；LAP 2606.03755（InstrumentCard、独占预约、签名安全确认，兼容 A2A/MCP，封装 SiLA 2/OPC-UA）；Bluesky；Lightfall 2606.06711（LLM 可寻址 beamline 控制） | 协议层没有位置，且 MHS 本身拿不到 |
| 艺术品自适应采样 | SLADS 用于 XRF 重建 1812.10836；自主自适应高光谱采集（Comm Bio 2020）；LightBot 机械臂自适应多光位采集（2022）；「高速/低剂量多目标自主扫描」专利 | 方法层 covered |
| 显微 agent | EAA 2602.15294（VLM 驱动显微工作流）；Janelia 案例 | covered |
| 音像迁移 | NOA MediaLector 8 路并行+纠错状态同步标注；Memnon/IU MDPI 百万级；磁带表面视频异常检测（CEUR 3865） | commercialized |
| 数字化 QC | Image Access「Quality Controlled Scanning」；FADGI 脚本化检查；Treventus ScanRobot | commercialized |
| 无人机考古 | 2025 Mimbres GPR+磁力仍人工飞；LLM 飞行规划 2607.06964（通用） | 通用层 covered，遗产层无人做但也无法接触硬件 |

## 判断二：仍未解决
- **遗产科学实验室的接入瓶颈是真的**（同 Janelia），文献里零 LLM-agent 先例（唯一遗产+LLM 实验室论文 2602.16551 是文献抽取，不碰仪器）。但缺口是**采用**，不是方法——按〈三〉属于 domain/product 贡献。
- **对象侧安全包络不存在**。MHS 记录的是**设备**安全限（重量、限位）；遗产场景里安全关键物是**对象**——每件对象的 lux·h 预算（JNF/100 年、PAS 198、微褪色测试结果）今天只在纸面与馆藏系统里，没有机器可读、跨仪器强制的形式。RIT 高光谱论文的光照是人工「按一小时展陈剂量」定的。
- **物理直觉失效在遗产里不可逆**：Genentech 的气泡可以重跑，一幅画的褪色不能。这就是保护伦理「最小干预」会挡住原件上自主执行的根本原因。

## 判断三：仍可能差异化（条件苛刻）
可证伪的问题：**给定每件对象的光/UV 剂量预算，agent 能否选择采集参数（波段、光位、张数）在最小剂量下达到指定可读性目标（例：印记/铭文可转写）并像 UW qPCR 那样择时停机？**
- 最近先例都缺一角：LightBot 有自适应几何无剂量约束；多光谱 RTI 框架无 agent 无预算；低剂量自主扫描在电镜/材料里有，无遗产对象语义。
- 诚实定位：**领域转置 + 对象预算语义**，落点是 JOCCH / Heritage Science / DAACH，不是 CCF-A。
- 成立条件（用户今天一项都不满足）：真实仪器（DIY RTI 灯罩+Raspberry Pi 数百镑可行）、有真实转写需求的对象类（拓片印记、风化碑刻、褪色手稿）、提供预算语义的保护方合作者。与 thesis 主干（VLM 评测/馆方阐释文本）正交。

## 最便宜的零更新基线
MCP server 包一层厂商 SDK + 固定采集协议（CHI 标准 RTI 光位）。agent 只有在采集本身「贵」（剂量或时间）时才有增量。

## 建议
Kill（作为本人 2027 前的论文线）。Reframe 的可逆最小动作：提交 MHS waitlist（用户自己填）；对象安全包络写成 note/position；RTI 灯罩 spike 仅在找到保护方合作者后再开。

Sources: MHS 公告 https://www.anthropic.com/news/model-hardware-standard-research-preview · LAP https://arxiv.org/abs/2606.03755 · Lightfall https://arxiv.org/pdf/2606.06711 · SLADS-XRF https://arxiv.org/pdf/1812.10836 · 自主高光谱 https://www.nature.com/articles/s42003-020-01385-3 · LightBot https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9143819/ · EAA https://arxiv.org/pdf/2602.15294 · 遗产 LLM 文献抽取 https://arxiv.org/pdf/2602.16551 · 磁带异常检测 https://ceur-ws.org/Vol-3865/13_paper.pdf · NOA/Memnon https://www.prosoundweb.com/noa-audio-solutions-helps-memnon-archiving-services-achieve-one-half-million-digitized-recordings/ · IU MDPI https://mdpi.iu.edu/archive/facilities.php · Image Access QC https://cdn.imageaccess.de/downloads/product_manuals/FAQ/FAQ-Quality-Controlled-Scanning.pdf · 微褪色 FAQ https://www.microfading.com/microfading-faq.html · 博物馆光照指南综述 https://www.nature.com/articles/s40494-026-02547-y · RIT 低光高光谱 https://repository.rit.edu/theses/10192/ · 多光谱 RTI https://library.imaging.org/archiving/articles/19/1/12 · Mimbres 无人机 https://www.sphengineering.com/news/gpr-and-magnetometry-drone-surveys-reveal-unexplored-archaeological-features-at-remote-mimbres-site · LLM 飞行规划 https://arxiv.org/pdf/2607.06964
