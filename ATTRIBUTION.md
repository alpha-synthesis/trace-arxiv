# Attribution of third-party content

## Wikipedia

The news contexts contain text from the day pages of the Wikipedia Current Events Portal ("Portal:Current events"), written by Wikipedia contributors and licensed under the Creative Commons Attribution-ShareAlike 4.0 license (https://creativecommons.org/licenses/by-sa/4.0/). Each day page was pinned to a fixed revision. The revision identifiers and timestamps are listed in `cache/wikipedia_pinned_manifest.json` (data bundle).

Changes made to the text: items were de-duplicated and truncated to 500 characters, and items whose sentences report stock-market or price moves were removed.

Files that contain this text, and are therefore shared under CC BY-SA 4.0:

- `cache/contexts_v2.1/` (data bundle)
- `experiments/audit/human_context_audit.xlsx`
- `experiments/audit/human_context_audit_sample.json`

## Market data

No market data is included. `tools/download_prices.py` downloads prices from Yahoo Finance through `yfinance`. Their use is subject to Yahoo's terms.

## Language models

The labels were produced by GPT-4o-mini (OpenAI) and, for the cross-model check, Claude Haiku 4.5 (Anthropic) and Gemini 2.5 Flash (Google), with Gemini 2.0 Flash labels for three stocks, all accessed through OpenRouter. The free-text fields of the labels (descriptions and raw responses) were emptied before release. Factor names, directions and magnitudes are kept.
