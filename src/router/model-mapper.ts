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

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

/**
 * Custom model mappings loaded from .router-mappings.json
 */
interface CustomMappings {
  [openaiModel: string]: string;
}

/**
 * Load custom model mappings from .router-mappings.json if it exists
 */
function loadCustomMappings(): CustomMappings {
  const mappingPath = join(process.cwd(), '.router-mappings.json');

  if (existsSync(mappingPath)) {
    try {
      const content = readFileSync(mappingPath, 'utf-8');
      return JSON.parse(content);
    } catch (error) {
      console.warn('Warning: Failed to load .router-mappings.json:', error);
      return {};
    }
  }

  return {};
}

/**
 * Check if model matches high-tier o-series pattern (o1, o3, o4 but not -mini variants)
 */
function isHighTierOSeries(model: string): boolean {
  // Match o1, o3, o4, etc. but exclude -mini variants
  const oSeriesMatch = model.match(/^o(\d+)/);
  if (oSeriesMatch && !model.includes('-mini')) {
    return true;
  }
  return false;
}

/**
 * Map OpenAI model name to Anthropic model name using pattern-based detection
 *
 * Mapping strategy:
 * 1. Check user's custom mappings file first (.router-mappings.json)
 * 2. Check environment variable override (ANTHROPIC_DEFAULT_MODEL)
 * 3. Pattern-based tier detection:
 *    - High-tier patterns (o1, o3, -pro, -max, -ultra) → claude-opus-4-5
 *    - Low-tier patterns (nano, gpt-3.5, gpt-3) → claude-haiku-4-5
 *    - Everything else → claude-sonnet-4-5 (default for MAX Plan)
 *
 * @param modelName OpenAI model name (e.g., 'gpt-4', 'gpt-5.2-pro', 'o3')
 * @returns Anthropic model name (e.g., 'claude-sonnet-4-5')
 */
export function mapOpenAIModelToAnthropic(modelName: string): string {
  // 1. Check custom mappings first
  const customMappings = loadCustomMappings();
  if (customMappings[modelName]) {
    return customMappings[modelName];
  }

  // 2. Check environment variable override
  if (process.env.ANTHROPIC_DEFAULT_MODEL) {
    return process.env.ANTHROPIC_DEFAULT_MODEL;
  }

  // 3. Pass through Anthropic model names unchanged
  if (modelName.startsWith('claude-')) {
    return modelName;
  }

  // 4. Pattern-based tier detection
  const model = modelName.toLowerCase();

  // High-tier patterns → Opus (premium/reasoning models)
  const highTierSuffixes = ['-pro', '-max', '-ultra'];
  if (highTierSuffixes.some((suffix) => model.includes(suffix))) {
    return 'claude-opus-4-5';
  }

  // High-tier o-series (o1, o3, etc. but not -mini variants)
  if (isHighTierOSeries(model)) {
    return 'claude-opus-4-5';
  }

  // Low-tier patterns → Haiku (fast/cheap models only)
  const lowTierPatterns = ['-nano', 'gpt-3.5', 'gpt-3'];
  if (lowTierPatterns.some((pattern) => model.includes(pattern))) {
    return 'claude-haiku-4-5';
  }

  // All other models → Sonnet (best balance for MAX Plan)
  // This includes: gpt-4, gpt-5, gpt-5.2, -mini, -turbo, etc.
  return 'claude-sonnet-4-5';
}

/**
 * Get a description of which pattern matched for a given model
 * Used for logging/debugging
 */
export function getModelMappingReason(modelName: string): string {
  const customMappings = loadCustomMappings();
  if (customMappings[modelName]) {
    return 'custom mapping';
  }

  if (process.env.ANTHROPIC_DEFAULT_MODEL) {
    return 'environment variable override';
  }

  if (modelName.startsWith('claude-')) {
    return 'anthropic model passthrough';
  }

  const model = modelName.toLowerCase();

  // Check high-tier patterns
  const highTierSuffixes = ['-pro', '-max', '-ultra'];
  if (highTierSuffixes.some((suffix) => model.includes(suffix))) {
    return 'high-tier pattern match';
  }

  if (isHighTierOSeries(model)) {
    return 'high-tier o-series match';
  }

  // Check low-tier patterns
  const lowTierPatterns = ['-nano', 'gpt-3.5', 'gpt-3'];
  if (lowTierPatterns.some((pattern) => model.includes(pattern))) {
    return 'low-tier pattern match';
  }

  return 'default pattern';
}

/**
 * Get all custom mappings for display purposes
 */
export function getCustomMappings(): CustomMappings {
  return loadCustomMappings();
}
