# Glossary

This glossary defines key terms and concepts used throughout LOCCA documentation.

## A

**Agent Loop**: The core reasoning cycle of observe-plan-act-reflect that drives intelligent behavior.

**Agent Patterns**: Standardized approaches for implementing AI agents, including frame-based and graph-based patterns.

## C

**CLI**: Command Line Interface - the primary way to interact with LOCCA through terminal commands.

**Configuration Layers**: LOCCA's three-tier configuration system (global, session, call) with precedence rules.

**Contracts**: Data schemas and interfaces that define how components communicate using Pydantic models.

## D

**Data Flow**: The path data takes through LOCCA components from user input to final response.

## L

**LangGraph**: A framework for building complex AI workflows using graph-based orchestration.

**LLM**: Large Language Model - AI models like GPT-4, Claude, or local models used for text generation.

## P

**Provider**: An LLM service provider (OpenAI, Anthropic, etc.) with routing, fallback, and health monitoring.

**Pydantic**: A Python library for data validation and parsing using type hints and models.

## R

**Runtime Manager**: Component responsible for session management, context persistence, and execution orchestration.

## S

**Sandbox**: Isolated execution environment for running tools securely with resource limits and security policies.

**Session**: A persistent context for maintaining conversation history and user preferences across interactions.

## T

**Tool System**: Extensible framework for agents to perform actions beyond LLM capabilities, with JSON schema validation.

**Three-Layer Configuration**: LOCCA's configuration hierarchy allowing global defaults, session overrides, and call-level customizations.
