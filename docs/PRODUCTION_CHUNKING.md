# Production Chunking Recommendation

Equal word splitting is not the preferred production strategy for real agent
traces. Use event-aware chunking.

## Parse the trace into events

Examples:

```text
system message
user prompt
assistant message or reasoning record
tool call
tool result
assistant final response
```

## Recommended algorithm

1. Parse structured events in sequence.
2. Preserve complete events whenever possible.
3. Estimate tokens with the tokenizer for the evaluator model.
4. Pack consecutive complete events until the target token budget is reached.
5. Carry the immediately previous event, or a bounded summary of it, into the
   next chunk as context.
6. Split inside an event only when one event alone exceeds the budget.
7. For an oversized event, split at sentence or token boundaries—not arbitrary
   character offsets.
8. Preserve event IDs, roles, timestamps, tool-call IDs, and sequence numbers.
9. Evaluate every chunk into strict structured JSON.
10. Group all chunk JSON records by trace/run ID.
11. Send those compact JSON records to the final reducer for one trace-level
    evaluation.

## Why this is better

It avoids separating:

- a user request from the corresponding assistant answer;
- a tool call from its tool result;
- an assistant claim from the evidence immediately around it.

A small overlap is useful, but overlap must be explicit metadata so downstream
systems can distinguish unique core content from copied context.
