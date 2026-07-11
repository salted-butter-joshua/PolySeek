# ADR-0006: 强类型配置（pydantic）+ 环境变量覆盖

- **状态**：Accepted
- **日期**：2026-07-10

## 背景与问题

原始设计到处传裸 `dict` 配置。字段拼错、类型错误要到运行中途才暴露；容器化又需要用环境
变量覆盖配置。如何做到启动即校验、部署可覆盖？

## 候选方案

- **方案 A**：继续传 `dict`。
- **方案 B**：pydantic `BaseSettings` 强类型模型 + YAML 源 + 环境变量源。

## 决策

采用方案 B：`AppConfig`（pydantic-settings），优先级 **环境变量 > YAML > 默认值**。

## 理由

1. 非法配置在加载阶段即报错（fail-fast），而非运行中途。
2. 上层拿到强类型对象，IDE 有补全，重构安全。
3. 容器部署用 `POLYSEEK__EMBEDDING__DEVICE=cuda` 覆盖，无需改文件。

## 后果

- **正面**：类型安全、可校验、12-factor 友好。
- **负面**：多一层模型定义；YAML 源须显式 UTF-8（否则 Windows 用 GBK 解码中文配置会崩）。
- **实现要点**：`settings_customise_sources` 里把 `YamlConfigSettingsSource(..., yaml_file_encoding="utf-8")`
  放在 env 源之后，实现 env > yaml 的优先级。
