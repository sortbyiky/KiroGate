# -*- coding: utf-8 -*-

# KiroGate
# Based on kiro-openai-gateway by Jwadow (https://github.com/Jwadow/kiro-openai-gateway)
# Original Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
OpenAI 兼容 API 的 Pydantic 模型。

Определяет схемы данных для запросов и ответов,
обеспечивая валидацию и сериализацию.
"""

import time
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field


# ==================================================================================================
# Модели для /v1/models endpoint
# ==================================================================================================

class OpenAIModel(BaseModel):
    """
    Модель данных для описания AI модели в формате OpenAI.
    
    Используется в ответе эндпоинта /v1/models.
    """
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "anthropic"
    description: Optional[str] = None


class ModelList(BaseModel):
    """
    Список моделей в формате OpenAI.
    
    Ответ эндпоинта GET /v1/models.
    """
    object: str = "list"
    data: List[OpenAIModel]


# ==================================================================================================
# Модели для /v1/chat/completions endpoint
# ==================================================================================================

class ChatMessage(BaseModel):
    """
    Сообщение в чате в формате OpenAI.
    
    Поддерживает различные роли (user, assistant, system, tool)
    и различные форматы контента (строка, список, объект).
    
    Attributes:
        role: Роль отправителя (user, assistant, system, tool)
        content: Содержимое сообщения (может быть строкой, списком или None)
        name: Опциональное имя отправителя
        tool_calls: Список вызовов инструментов (для assistant)
        tool_call_id: ID вызова инструмента (для tool)
    """
    role: str
    content: Optional[Union[str, List[Any], Any]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None
    
    model_config = {"extra": "allow"}


class ToolFunction(BaseModel):
    """
    Описание функции инструмента.
    
    Attributes:
        name: Имя функции
        description: Описание функции
        parameters: JSON Schema параметров функции
    """
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    """
    Инструмент (tool) в формате OpenAI.
    
    Attributes:
        type: Тип инструмента (обычно "function")
        function: Описание функции
    """
    type: str = "function"
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    """
    Запрос на генерацию ответа в формате OpenAI Chat Completions API.
    
    Поддерживает все стандартные поля OpenAI API, включая:
    - Базовые параметры (model, messages, stream)
    - Параметры генерации (temperature, top_p, max_tokens)
    - Tools (function calling)
    - Дополнительные параметры (игнорируются, но принимаются для совместимости)
    
    Attributes:
        model: ID модели для генерации
        messages: Список сообщений чата
        stream: Использовать streaming (по умолчанию False)
        temperature: Температура генерации (0-2)
        top_p: Top-p sampling
        n: Количество вариантов ответа
        max_tokens: Максимальное количество токенов в ответе
        max_completion_tokens: Альтернативное поле для max_tokens
        stop: Стоп-последовательности
        presence_penalty: Штраф за повторение тем
        frequency_penalty: Штраф за повторение слов
        tools: Список доступных инструментов
        tool_choice: Стратегия выбора инструмента
    """
    model: str
    messages: Annotated[List[ChatMessage], Field(min_length=1)]
    stream: bool = False
    
    # Параметры генерации
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    
    # Tools (function calling)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    
    # Поля для совместимости (игнорируются)
    stream_options: Optional[Dict[str, Any]] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    
    model_config = {"extra": "allow"}


# ==================================================================================================
# Модели для ответов
# ==================================================================================================

class ChatCompletionChoice(BaseModel):
    """
    Один вариант ответа в Chat Completion.
    
    Attributes:
        index: Индекс варианта
        message: Сообщение ответа
        finish_reason: Причина завершения (stop, tool_calls, length)
    """
    index: int = 0
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """
    Информация об использовании токенов.
    
    Attributes:
        prompt_tokens: Количество токенов в запросе
        completion_tokens: Количество токенов в ответе
        total_tokens: Общее количество токенов
        credits_used: Использованные кредиты (специфично для Kiro)
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    credits_used: Optional[float] = None


class ChatCompletionResponse(BaseModel):
    """
    Полный ответ Chat Completion (non-streaming).
    
    Attributes:
        id: Уникальный ID ответа
        object: Тип объекта ("chat.completion")
        created: Timestamp создания
        model: Использованная модель
        choices: Список вариантов ответа
        usage: Информация об использовании токенов
    """
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(BaseModel):
    """
    Дельта изменений в streaming chunk.
    
    Attributes:
        role: Роль (только в первом chunk)
        content: Новый контент
        tool_calls: Новые tool calls
    """
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionChunkChoice(BaseModel):
    """
    Один вариант в streaming chunk.
    
    Attributes:
        index: Индекс варианта
        delta: Дельта изменений
        finish_reason: Причина завершения (только в последнем chunk)
    """
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """
    Streaming chunk в формате OpenAI.

    Attributes:
        id: Уникальный ID ответа
        object: Тип объекта ("chat.completion.chunk")
        created: Timestamp создания
        model: Использованная модель
        choices: Список вариантов
        usage: Информация об использовании (только в последнем chunk)
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[ChatCompletionUsage] = None


# ==================================================================================================
# Модели для Anthropic Messages API (/v1/messages)
# ==================================================================================================

class AnthropicContentBlock(BaseModel):
    """
    Content block для Anthropic Messages API.

    Поддерживает различные типы контента: text, image, tool_use, tool_result, thinking.

    Attributes:
        type: Тип контента
        text: Текстовое содержимое (для type="text")
        source: Источник изображения (для type="image")
        id: ID tool_use (для type="tool_use")
        name: Имя инструмента (для type="tool_use")
        input: Входные данные инструмента (для type="tool_use")
        tool_use_id: ID связанного tool_use (для type="tool_result")
        content: Результат инструмента (для type="tool_result")
        is_error: Флаг ошибки (для type="tool_result")
        thinking: Содержимое thinking (для type="thinking")
    """
    type: str  # "text", "image", "tool_use", "tool_result", "thinking"
    text: Optional[str] = None
    # image fields
    source: Optional[Dict[str, Any]] = None  # {"type": "base64"/"url", "media_type": "...", "data"/"url": "..."}
    # tool_use fields
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # tool_result fields
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List[Any]]] = None
    is_error: Optional[bool] = None
    # thinking fields
    thinking: Optional[str] = None

    model_config = {"extra": "allow"}


class AnthropicMessage(BaseModel):
    """
    Сообщение в формате Anthropic.

    Attributes:
        role: Роль (user или assistant)
        content: Содержимое (строка или список content blocks)
    """
    role: str  # "user" or "assistant"
    content: Union[str, List[AnthropicContentBlock], List[Dict[str, Any]]]

    model_config = {"extra": "allow"}


class AnthropicTool(BaseModel):
    """
    Инструмент в формате Anthropic.

    Attributes:
        name: Имя инструмента
        description: Описание инструмента
        input_schema: JSON Schema входных параметров
    """
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

    model_config = {"extra": "allow"}


class AnthropicMessagesRequest(BaseModel):
    """
    Запрос к Anthropic Messages API.

    Attributes:
        model: ID модели
        messages: Список сообщений
        max_tokens: Максимальное количество токенов (обязательно)
        system: Системный промпт
        tools: Список инструментов
        tool_choice: Стратегия выбора инструмента
        temperature: Температура генерации
        top_p: Top-p sampling
        top_k: Top-k sampling
        stop_sequences: Стоп-последовательности
        stream: Использовать streaming
        metadata: Метаданные запроса
        thinking: Настройки extended thinking
    """
    model: str
    messages: Annotated[List[AnthropicMessage], Field(min_length=1)]
    max_tokens: int  # Required in Anthropic API
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None
    # Extended Thinking support
    thinking: Optional[Dict[str, Any]] = None  # {"type": "enabled", "budget_tokens": 1024}

    model_config = {"extra": "allow"}


class AnthropicUsage(BaseModel):
    """
    Информация об использовании токенов в формате Anthropic.

    Attributes:
        input_tokens: Количество входных токенов
        output_tokens: Количество выходных токенов
    """
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponseContentBlock(BaseModel):
    """
    Content block в ответе Anthropic.

    Attributes:
        type: Тип контента (text, tool_use, thinking)
        text: Текстовое содержимое
        id: ID tool_use
        name: Имя инструмента
        input: Входные данные инструмента
        thinking: Содержимое thinking
    """
    type: str  # "text", "tool_use", "thinking"
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    thinking: Optional[str] = None


class AnthropicMessagesResponse(BaseModel):
    """
    Ответ Anthropic Messages API.

    Attributes:
        id: Уникальный ID ответа
        type: Тип объекта (всегда "message")
        role: Роль (всегда "assistant")
        content: Список content blocks
        model: Использованная модель
        stop_reason: Причина остановки
        stop_sequence: Сработавшая стоп-последовательность
        usage: Информация об использовании токенов
    """
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicResponseContentBlock]
    model: str
    stop_reason: Optional[str] = None  # "end_turn", "max_tokens", "tool_use", "stop_sequence"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage