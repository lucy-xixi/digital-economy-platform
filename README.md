# Digital Economy Intelligence Platform (DEIP)

面向政府研究院与企业战略部门的**数字经济情报分析平台** Demo。  

---


| 维度 | 企业级信号 | 本仓库落点 |
|------|------------|------------|
| 数据工程 | 契约、硬门禁、幂等入库、run 血缘、CLI | `contracts/` · `QualityGate` · `DeipRepository` upsert · `PipelineContext` · `python -m src.cli` |
| 统计分析 | 面板 within 相关、OLS 区间、回测、残差诊断、权重敏感性、**因果效应** | `correlation` · `forecasting` · `diagnostics` · `sensitivity` · **`causal_analysis`** |
| AI 应用 | 工具调用、结构化输出、轻量 RAG、grounding 校验 | `ai_agent/tools.py` · `ResearchBrief` · `docs/methodology.md` |
| 工程完备 | 类型提示、日志、Docker、pytest 不变量 | `tests/test_pipeline.py` · `Dockerfile` |

---

## 架构

```
CLI / Streamlit
      │
      ▼
PipelineContext(run_id) ──► lineage/<run_id>.json
      │
      ├─ ingest ── contract(raw)
      ├─ validate ── HARD GATE ──► data_quality_reports
      ├─ clean ── contract(clean)
      ├─ compute DEI ── contract(dei)
      ├─ upsert SQLite (ON CONFLICT, transactional)
      └─ analytics warm-up (panel / OLS-PI / backtest / sensitivity)
                │
                ▼
        Tool-grounded Research Agent
        (trend · corr · forecast · diagnostics · sensitivity · methodology)
```

**DEI 权重：** Infrastructure 30% · Industry 30% · AI Innovation 20% · Application 20%

---

## 快速开始

```bash
cd digital-economy-platform
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.cli run                 # 全链路 + 门禁 + 血缘
python -m src.cli validate            # 仅质检门禁
python -m src.cli forecast            # OLS 预测区间 + walk-forward
python -m src.cli causal              # AI→DEI 因果效应 + 置信区间
python -m src.cli report "分析未来数字经济趋势"
streamlit run dashboard/app.py
pytest tests/ -v
```

Docker：`docker compose up --build`

---

## 关键设计说明

### 数据工程
- **Contract `metrics_v1`**：raw / clean / DEI 三阶段列、唯一键、值域校验；失败抛 `ContractViolationError`
- **Hard DQ Gate**：分数、缺失率、重复键、地区覆盖不达标则 `QualityGateError` 阻断下游
- **Upsert**：`INSERT … ON CONFLICT DO UPDATE`，写入绑定 `run_id`（告别 truncate-reload 叙事）
- **Lineage**：阶段耗时、产物 SHA256、config_hash → `data/processed/lineage/`

### 统计分析
- **Pooled vs Within** Pearson（地区去均值，接近固定效应直觉）
- **statsmodels OLS** + **95% prediction interval** + Durbin–Watson
- **Walk-forward** 一步超前 MAE
- **Jarque–Bera / Breusch–Pagan** 残差诊断
- **权重 ±10%** 敏感性 → Spearman 排名稳定性
- **因果分析**：DoWhy backdoor（AI Innovation → DEI）+ 置信区间；`dei_ex_ai` 溢出稳健性；可选 EconML LinearDML

### AI
- Agent 先跑 **tools**，再组装 **`ResearchBrief`（Pydantic）**
- Mock 路径仍引用真实工具数字（可 eval）
- `retrieve_methodology`：基于 `docs/methodology.md` 的关键词检索（可审计的轻量 RAG）
- Grounding：关键数字须出现在 findings evidence

---


## License

MIT — Demo / educational use.
