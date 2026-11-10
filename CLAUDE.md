Project rules
Context

Interview prototype: multi-agent retail banking customer assistant. Everything AI is real (Mistral API); everything bank-side is mocked behind the interfaces a bank would expose. The owner must be able to explain every line. Full spec: SPEC.md.

Secrets
A .env file exists with MISTRAL_API_KEY. Never read, print, create, overwrite, move or delete it. Never echo environment variables.
Check it exists without reading it: test -s .env && echo ok. If missing, stop and tell me.
No key in code, logs, test output or UI.
Working rules
Work stage by stage as asked. Do not start the next stage without my go.
Before coding a stage, list ambiguities and ask.
After each stage: run pytest -v --tb=short and show the REAL complete output (never a summary written by you). All tests pass, none skipped.
Never run git commit or git push: I commit myself.
Do not start the app (python app.py): I start it myself.
Keep code short and plain: functions, no classes unless needed, no agent framework (no LangChain, LlamaIndex, etc.), no features beyond SPEC.md, no unused code. Comment every function in one or two lines.
Use the current mistralai SDK syntax that is installed (check the installed version before writing calls; never the old MistralClient / ChatMessage syntax).
When done with a stage: list files changed and explain in 5 lines what each new function does.

Current state (handover)
Done: stages 1 to 6 (mocked bank APIs and audit, FAQ documents and RAG, guardrails and router, FAQ/account/action/handoff agents with confirm/cancel, Gradio app, eval.py + data/golden_set.json + README) plus the fixes from manual testing listed in docs/BUGS.md and the rules in SPEC.md "Decisions v1.2" (help route, visitor mode, courtesy words, sticky language, card identification in code, pending action cleared by any new message). 86 tests pass.
Eval (python eval.py, 32 golden cases covering every route in EN and FR and every regression from docs/BUGS.md): routing 32/32, checks 66/66, 0 API errors. Latency confirmed: output moderation skipped on fixed templates, input guardrails in parallel with the router. Median turn latency: faq 2.6 s, account 2.8 s, everything else under about 2 s.
Not committed at handover: stage 6 (eval.py, data/golden_set.json, README.md, .gitignore). The owner commits.
Left:
- Decide whether faq and account latency (p90 about 4-5 s) needs more work.
Read docs/BUGS.md before changing agents.py.

