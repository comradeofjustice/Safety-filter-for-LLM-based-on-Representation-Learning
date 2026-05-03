# 4models/REVIEW_LOG.md — 多 Worker 公共协调日志
# 所有人 append-only，写完 commit

---

[2026-05-04 01:00] worker=A status=DONE task=R1 (no-op on code)
  - 已通知 D 修正 §4.1 "70% of evaluation data" 笔误
  - 现有 models/deepsafe_v3_8B/、embeddings/qwen3-embedding-8B/、
    valuation/heldout/predictions/ 全部继续使用，不做任何数据重切或模型重训

[REQUEST] from=A to=D: §4.1 "70% of evaluation data" 是笔误，
  请改成 "training set draws only from WildGuardMix/train and
  AegisAI-Content-Safety-2.0/train (with optional 100PoisonMpts/train
  and XGuard-Train-Open-200K); all 9 evaluation benchmarks
  (AegisAI-1.0, AegisAI-2.0/test, BeaverTails, CRiskEval, DoNotAnswer,
  MM-SafetyBench, SafetyBench, ToxicChat, XSTest) are 100% held-out."
  附注: 这是纯文字修正，不涉及代码或数据变更。所有既有模型和预测结果仍然有效。

[2026-05-04 12:30] worker=D status=DONE tasks=R1,R9,R10  bulk=text-only
  - R1 DONE: §4.1 "70% of evaluation data" 改为正确训练/评估分离描述
  - R9 DONE: 新增 4 条引用 (Ganea2018 hyperbolic NN, Snell2017 proto nets,
    Zhang2018 mixup, Guo2017 calibration) 并在 §3.2, §3.6, §3.3 引用
  - R10a DONE: checklist "Section~5/6" → "Section~4/5"
  - R10b DONE: 门控标量 α→β, 对比系数保留 α
  - R10c DONE: §3.7 Eq.(13) 展开显式列出 λ_inter=0.5
  - R10d DONE: Abstract 参数量改为 "7.3M-parameter trainable head atop frozen 8B encoder"
  - R10e DONE: §4.4 删除 "40×" 改为 silhouette ≈0 → ≈0.30
  - R10f DONE: Table 2 caption 注明 validation split; Figure 4 caption 添加 9×9=81 pairings
  - R10g DONE: Conclusion "directly win against" → "match or exceed"
  - Created paper/inject_numbers.py → paper/auto_numbers.tex (60 LaTeX macros)
  - Cleaned paper/paper/figures/ historical duplicate directory
  - main.tex compiles clean: 0 errors, 13 pages
  - All number macros ready for CSV-driven injection once A/B/C produce reports/*.csv

[REQUEST] from=D to=B: R6 counterfactual 文字待定。
  请确认是否保留 counterfactual pair repulsion loss。
  若保留, D 将在 §3.7 总损失里对齐公式;
  若删除, D 将全文清理 counterfactual 相关内容。
  当前状态: 保持原文不变,等待 B 指令。

[REQUEST] from=D to=A,B,C: reports/ 下的 main_table.csv / ablation_table.csv /
  stats_with_ci.csv / baselines_table.csv 尚不存在。
  请尽快产出这些 CSV (格式见 paper/inject_numbers.py 表头注释),
  否则正文数字仍使用硬编码占位值。

