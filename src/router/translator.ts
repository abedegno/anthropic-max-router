/**
 * EDUCATIONAL AND ENTERTAINMENT PURPOSES ONLY
 *
 * This software is provided for educational, research, and entertainment purposes only.
 * It is not affiliated with, endorsed by, or sponsored by Anthropic PBC.
 * Use at your own risk. No warranties provided. Users are solely responsible for
 * ensuring compliance with Anthropic's Terms of Service and all applicable laws.
 *
 * Copyright (c) 2025 - Licensed under MIT License
 */

import {
  OpenAIChatCompletionRequest,
  OpenAIMessage,
  OpenAIContentBlock,
  OpenAITool,
  OpenAIChatCompletionResponse,
  OpenAIErrorResponse,
  AnthropicRequest,
  AnthropicResponse,
  Message,
  Tool,
  ContentBlock,
} from '../types.js';
import { mapOpenAIModelToAnthropic } from './model-mapper.js';

/**
 * Extract text from OpenAI message content, which can be a string, null,
 * or an array of content blocks per the OpenAI Chat Completions spec.
 */
function extractTextContent(content: string | OpenAIContentBlock[] | null): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((block) => block.type === 'text' && block.text)
      .map((block) => block.text!)
      .join('');
  }
  return '';
}

/**
 * Translate OpenAI Chat Completion request to Anthropic Messages API request
 */
export function translateOpenAIToAnthropic(
  openaiRequest: OpenAIChatCompletionRequest
): AnthropicRequest {
  // Extract and combine all system messages
  const systemMessages: string[] = [];
  const conversationMessages: OpenAIMessage[] = [];

  for (const msg of openaiRequest.messages) {
    if (msg.role === 'system') {
      systemMessages.push(extractTextContent(msg.content));
    } else {
      conversationMessages.push(msg);
    }
  }

  // Build Anthropic messages handling tool_use/tool_result round-trips
  const anthropicMessages: Message[] = [];

  for (const msg of conversationMessages) {
    if (msg.role === 'assistant') {
      // Build content blocks for assistant messages
      const contentBlocks: ContentBlock[] = [];
      const text = extractTextContent(msg.content);
      if (text) {
        contentBlocks.push({ type: 'text', text });
      }
      // Convert OpenAI tool_calls to Anthropic tool_use blocks
      if (msg.tool_calls && msg.tool_calls.length > 0) {
        for (const tc of msg.tool_calls) {
          let input: Record<string, unknown> = {};
          try {
            input = JSON.parse(tc.function.arguments);
          } catch {
            // Keep empty input if parse fails
          }
          contentBlocks.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.function.name,
            input,
          });
        }
      }
      if (contentBlocks.length > 0) {
        // Merge with previous assistant message if needed (Anthropic requires alternation)
        const prev = anthropicMessages[anthropicMessages.length - 1];
        if (prev && prev.role === 'assistant') {
          if (typeof prev.content === 'string') {
            prev.content = [{ type: 'text', text: prev.content }];
          }
          (prev.content as ContentBlock[]).push(...contentBlocks);
        } else {
          anthropicMessages.push({
            role: 'assistant',
            content:
              contentBlocks.length === 1 && contentBlocks[0].type === 'text'
                ? (contentBlocks[0].text as string)
                : contentBlocks,
          });
        }
      }
    } else if (msg.role === 'tool') {
      // Convert OpenAI tool result to Anthropic tool_result in a user message
      const toolResultBlock: ContentBlock = {
        type: 'tool_result',
        tool_use_id: msg.tool_call_id,
        content: extractTextContent(msg.content) || '',
      };
      // Anthropic tool_results must be in user messages
      const prev = anthropicMessages[anthropicMessages.length - 1];
      if (prev && prev.role === 'user' && Array.isArray(prev.content)) {
        (prev.content as ContentBlock[]).push(toolResultBlock);
      } else {
        anthropicMessages.push({
          role: 'user',
          content: [toolResultBlock],
        });
      }
    } else {
      // Regular user message
      const text = extractTextContent(msg.content);
      const prev = anthropicMessages[anthropicMessages.length - 1];
      if (prev && prev.role === 'user') {
        // Merge consecutive user messages
        if (typeof prev.content === 'string') {
          prev.content = prev.content + '\n\n' + text;
        } else if (Array.isArray(prev.content)) {
          (prev.content as ContentBlock[]).push({ type: 'text', text });
        }
      } else {
        anthropicMessages.push({ role: 'user', content: text });
      }
    }
  }

  // Translate tools if present
  let anthropicTools: Tool[] | undefined;
  if (openaiRequest.tools && openaiRequest.tools.length > 0) {
    anthropicTools = openaiRequest.tools.map(translateOpenAIToolToAnthropic);
  }

  // Build the Anthropic request
  const anthropicRequest: AnthropicRequest = {
    model: mapOpenAIModelToAnthropic(openaiRequest.model),
    max_tokens: openaiRequest.max_tokens || 16384,
    messages: anthropicMessages,
    stream: openaiRequest.stream || false,
  };

  // Add system messages if present
  if (systemMessages.length > 0) {
    anthropicRequest.system = [
      {
        type: 'text',
        text: systemMessages.join('\n\n'),
      },
    ];
  }

  // Add tools if present
  if (anthropicTools && anthropicTools.length > 0) {
    anthropicRequest.tools = anthropicTools;
  }

  return anthropicRequest;
}

/**
 * Translate OpenAI tool to Anthropic tool
 */
function translateOpenAIToolToAnthropic(openaiTool: OpenAITool): Tool {
  return {
    name: openaiTool.function.name,
    description: openaiTool.function.description,
    input_schema: {
      type: 'object',
      properties: openaiTool.function.parameters.properties,
      required: openaiTool.function.parameters.required,
    },
  };
}

/**
 * Translate Anthropic response to OpenAI Chat Completion response
 */
export function translateAnthropicToOpenAI(
  anthropicResponse: AnthropicResponse,
  originalModel: string
): OpenAIChatCompletionResponse {
  // Extract text content from Anthropic's content blocks
  const textBlocks = anthropicResponse.content.filter(
    (block: ContentBlock) => block.type === 'text'
  );
  const content = textBlocks.map((block: ContentBlock) => block.text).join('');

  // Map stop_reason to finish_reason
  let finishReason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | null = null;
  if (anthropicResponse.stop_reason === 'end_turn') {
    finishReason = 'stop';
  } else if (anthropicResponse.stop_reason === 'max_tokens') {
    finishReason = 'length';
  } else if (anthropicResponse.stop_reason === 'tool_use') {
    finishReason = 'tool_calls';
  }

  // Check if there are tool uses
  const toolUseBlocks = anthropicResponse.content.filter(
    (block: ContentBlock) => block.type === 'tool_use'
  );

  const toolCalls =
    toolUseBlocks.length > 0
      ? toolUseBlocks.map((block: ContentBlock) => ({
          id: block.id as string,
          type: 'function' as const,
          function: {
            name: block.name as string,
            arguments: JSON.stringify(block.input),
          },
        }))
      : undefined;

  return {
    id: anthropicResponse.id,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: originalModel,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: content || null,
          ...(toolCalls && { tool_calls: toolCalls }),
        },
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: anthropicResponse.usage.input_tokens,
      completion_tokens: anthropicResponse.usage.output_tokens,
      total_tokens: anthropicResponse.usage.input_tokens + anthropicResponse.usage.output_tokens,
    },
  };
}

/**
 * Translate Anthropic streaming events to OpenAI streaming format
 * This returns a generator that yields OpenAI-formatted SSE strings
 */
export async function* translateAnthropicStreamToOpenAI(
  anthropicStream: AsyncIterable<Uint8Array>,
  originalModel: string,
  messageId: string
): AsyncGenerator<string, void, unknown> {
  const decoder = new TextDecoder();
  let buffer = '';
  let totalInputTokens = 0;
  let totalOutputTokens = 0;

  // Track tool_use blocks being streamed
  const toolCalls: { index: number; id: string; name: string; arguments: string }[] = [];
  let currentToolIndex = -1;
  let currentToolArgs = '';
  let finishReason: string = 'stop';

  // Send initial chunk with role
  yield `data: ${JSON.stringify({
    id: messageId,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: originalModel,
    choices: [
      {
        index: 0,
        delta: { role: 'assistant' },
        finish_reason: null,
      },
    ],
  })}\n\n`;

  for await (const chunk of anthropicStream) {
    buffer += decoder.decode(chunk, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim() || line.startsWith(':')) continue;

      if (line.startsWith('data: ')) {
        const data = line.slice(6);

        try {
          const event = JSON.parse(data);

          if (
            event.type === 'content_block_start' &&
            event.content_block?.type === 'tool_use'
          ) {
            // Start of a tool_use block — flush any previous tool args
            if (currentToolIndex >= 0 && currentToolArgs) {
              toolCalls[currentToolIndex].arguments = currentToolArgs;
            }
            currentToolIndex = toolCalls.length;
            currentToolArgs = '';
            toolCalls.push({
              index: currentToolIndex,
              id: event.content_block.id,
              name: event.content_block.name,
              arguments: '',
            });
            // Send tool call start chunk
            yield `data: ${JSON.stringify({
              id: messageId,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: originalModel,
              choices: [
                {
                  index: 0,
                  delta: {
                    tool_calls: [
                      {
                        index: currentToolIndex,
                        id: event.content_block.id,
                        type: 'function',
                        function: { name: event.content_block.name, arguments: '' },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            })}\n\n`;
          } else if (
            event.type === 'content_block_delta' &&
            event.delta?.type === 'input_json_delta'
          ) {
            // Tool arguments streaming
            currentToolArgs += event.delta.partial_json;
            yield `data: ${JSON.stringify({
              id: messageId,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: originalModel,
              choices: [
                {
                  index: 0,
                  delta: {
                    tool_calls: [
                      {
                        index: currentToolIndex,
                        function: { arguments: event.delta.partial_json },
                      },
                    ],
                  },
                  finish_reason: null,
                },
              ],
            })}\n\n`;
          } else if (
            event.type === 'content_block_delta' &&
            event.delta?.type === 'text_delta'
          ) {
            // Text content delta
            yield `data: ${JSON.stringify({
              id: messageId,
              object: 'chat.completion.chunk',
              created: Math.floor(Date.now() / 1000),
              model: originalModel,
              choices: [
                {
                  index: 0,
                  delta: { content: event.delta.text },
                  finish_reason: null,
                },
              ],
            })}\n\n`;
          } else if (event.type === 'message_delta') {
            if (event.usage) {
              totalOutputTokens = event.usage.output_tokens || totalOutputTokens;
            }
            if (event.delta?.stop_reason === 'tool_use') {
              finishReason = 'tool_calls';
            } else if (event.delta?.stop_reason === 'max_tokens') {
              finishReason = 'length';
            } else if (event.delta?.stop_reason) {
              finishReason = 'stop';
            }
          } else if (event.type === 'message_start' && event.message?.usage) {
            // Initial token count
            totalInputTokens = event.message.usage.input_tokens || 0;
          }
        } catch {
          // Ignore parse errors for streaming events
        }
      }
    }
  }

  // Flush final tool args
  if (currentToolIndex >= 0 && currentToolArgs) {
    toolCalls[currentToolIndex].arguments = currentToolArgs;
  }

  // Send final chunk with usage information
  yield `data: ${JSON.stringify({
    id: messageId,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: originalModel,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: finishReason,
      },
    ],
    usage: {
      prompt_tokens: totalInputTokens,
      completion_tokens: totalOutputTokens,
      total_tokens: totalInputTokens + totalOutputTokens,
    },
  })}\n\n`;

  // Send [DONE] marker
  yield 'data: [DONE]\n\n';
}

/**
 * Translate Anthropic error to OpenAI error format
 */
export function translateAnthropicErrorToOpenAI(error: unknown): OpenAIErrorResponse {
  // If it's already an Anthropic error format, translate it
  const err = error as { error?: { type?: string; message?: string }; message?: string };
  if (err.error?.type && err.error?.message) {
    return {
      error: {
        message: err.error.message,
        type: err.error.type,
        param: null,
        code: null,
      },
    };
  }

  // Generic error
  return {
    error: {
      message: err.message || 'An error occurred',
      type: 'internal_error',
      param: null,
      code: null,
    },
  };
}

/**
 * Validate OpenAI request and throw errors for unsupported features
 */
export function validateOpenAIRequest(request: OpenAIChatCompletionRequest): void {
  // Error on unsupported features that would change behavior
  if (request.n && request.n > 1) {
    throw new Error(
      'Multiple completions (n > 1) are not supported. Anthropic only returns one completion.'
    );
  }

  if (request.logprobs) {
    throw new Error('Log probabilities (logprobs) are not supported by Anthropic API.');
  }

  // Warn about ignored parameters (these won't cause errors but won't work as expected)
  if (request.presence_penalty !== undefined) {
    console.warn('Warning: presence_penalty is not supported by Anthropic and will be ignored');
  }

  if (request.frequency_penalty !== undefined) {
    console.warn('Warning: frequency_penalty is not supported by Anthropic and will be ignored');
  }

  if (request.logit_bias !== undefined) {
    console.warn('Warning: logit_bias is not supported by Anthropic and will be ignored');
  }
}
