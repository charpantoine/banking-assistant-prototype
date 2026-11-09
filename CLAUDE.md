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