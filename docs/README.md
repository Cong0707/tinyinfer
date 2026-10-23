# 文档

## 入门

- [QUICKSTART](QUICKSTART.md) — 从零到跑起来
- [ENV](ENV.md) — 环境变量与运行时开关
- [HARDWARE](HARDWARE.md) — 硬件要求与实测机型
- [faq](faq.md) — 常见问题

## 设计

- [architecture](architecture.md) — 模块结构、流式模型、会话
- [PHILOSOPHY](PHILOSOPHY.md) — 取舍的理由
- [loading](loading.md) — 权重加载、分片、跨盘布局
- [level2_mechanisms](level2_mechanisms.md) — 流式、专家缓存、门控预取
- [mtp_design](mtp_design.md) — 多 token 预测
- [long_context](long_context.md) — 长上下文与压缩注意力

## Kernel

- [kernels](kernels.md) — 每个 kernel 的实现说明
- [kernels_provenance](kernels_provenance.md) — 官方移植 / 重实现的来源对照

## 性能

- [PERFORMANCE](PERFORMANCE.md) — 实测数字
- [benchmarks](benchmarks.md) — 基准方法与经验规则
- [decode_speed](decode_speed.md) — 每项优化带来的变化
- [lower_bounds](lower_bounds.md) — 硬件物理下限的推算

## 正确性

- [parity](parity.md) — 与参考实现的分量级对比
- [GLOSSARY](GLOSSARY.md) — 术语（slab / ring / slot / tier）

## 使用

- [api](api.md) — HTTP API
- [OPERATIONS](OPERATIONS.md) — 运维
- [EXTENDING](EXTENDING.md) — 接入新模型 / 新 kernel
- [changelog](changelog.md) — 变更记录
- [ROADMAP](ROADMAP.md) — 计划

## history/

过程记录与早期分析，保留备查，不是给使用者看的。
