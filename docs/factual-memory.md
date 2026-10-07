# Conversation facts and sources

Factual memory keeps original dialogue alongside summaries so later prompts can
retrieve attributed evidence. The implementation is in
[factual_memory.py](../agent/core/factual_memory.py), with graph integration in
[langgraph_builder.py](../agent/core/langgraph_builder.py) and prompt rendering in
[prompt_builder.py](../agent/core/prompt_builder.py).

Each therapist/patient exchange is appended to
`data/memory/<therapist_id>__<patient_id>.jsonl` before the five-turn history window
is trimmed. Records include session and turn identifiers. Safety-flagged turns
remain archived but are excluded from factual retrieval. Clinical profile fields
are not changed by extraction.

Session finalization creates narrative memory and extracts facts from
unprocessed source turns. A fact must identify its source and speaker, contain a
literal quote from that speaker, and place its value inside the quote. Guards
reject questions and unsupported agreement/completion claims. Patient and
therapist statements remain separately attributed; a proposal cannot replace an
agreement. Earlier versions remain available, ordered by source chronology.

Native finalization quarantines invalid facts or malformed nonempty extraction
responses and preserves their validation reasons. Accepted facts remain usable,
and processed batches are skipped on retry. Provider errors, empty responses or
storage failures propagate and keep the session open. Direct callers use strict
validation by default, rejecting an invalid batch atomically.

The API reports `memory_status="complete"` or `"partial"` with warnings when
extraction is incomplete. These statuses describe validation checks, not clinical
truth or exhaustive recall.

Retrieval combines lexical matching and local embeddings over facts and original
utterances, including turns not yet extracted. Prompts show attributed quotes
and current/superseded markers. Default budgets are eight evidence records at
approximately 1,800 tokens, recent dialogue at 1,800 and narrative context at
700. These are local estimates; quotes are included whole or omitted.

Literal validation establishes provenance, not full semantic entailment. Entity
selection and generated answers still depend on the model. Historical
narrative-only archives remain readable; missing original turns are not
reconstructed. See [offline validation](runtime-validation.md) for regression
checks using synthetic fixtures.
