# ABD3 developer shortcuts. Every target is a thin wrapper around an
# existing script/CLI — no hidden logic lives here.
#
# Expect CUDA_VISIBLE_DEVICES to be set by the caller when relevant.
# Use `make help` to list targets.

.PHONY: help install install-dev test test-fast smoke smoke-ddp feas ppl gen \
        zero-shot diversity gen-ppl mauve quality efficiency collect-results \
        lint format check clean

PY            ?= python
PYTEST        ?= $(PY) -m pytest
CKPT_FEAS     ?= checkpoints/feasibility/epoch=43-step=4000.ckpt
CKPT_BASELINE ?= checkpoints/ablation_baseline/epoch=33-step=3000.ckpt

help:  ## Show this help.
	@awk 'BEGIN {FS=":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*?##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install runtime deps.
	pip install -r requirements.txt

install-dev:  ## Install runtime + dev deps (ruff, pytest, pre-commit).
	pip install -r requirements.txt
	pip install pytest ruff==0.6.9 pre-commit==4.0.1

test:  ## Run the full unit-test suite.
	$(PYTEST) tests/ -v

test-fast:  ## Run unit tests, stop on first failure.
	$(PYTEST) tests/ -x -q

smoke:  ## CPU smoke test (1 step, block_size=4) — tiny sanity check.
	$(PY) -m abd3.main --config-name=feasibility \
		trainer.max_steps=1 trainer.devices=1 trainer.accelerator=cpu \
		loader.batch_size=1 loader.num_workers=0

smoke-ddp:  ## DDP smoke on all visible GPUs (5 steps).
	$(PY) -m abd3.main --config-name=feasibility \
		trainer.max_steps=5 loader.batch_size=4

feas:  ## Full 5k-step feasibility run (requires ~24GB GPU).
	bash scripts/run_all_experiments.sh

ppl:  ## Compute importance-weighted MC-ELBO PPL on the feasibility ckpt.
	$(PY) -m eval.perplexity --checkpoint $(CKPT_FEAS) \
		--config-name feasibility --split test --device cuda --use-ema

gen:  ## Generate 4 samples from the feasibility ckpt (semi-AR, block_size=4).
	$(PY) -m eval.generate --checkpoint $(CKPT_FEAS) \
		--n-samples 4 --num-steps 20 --block-size 4 --device cuda \
		--use-ema --out report/samples/smoke.jsonl --print-samples

zero-shot:  ## Zero-shot PPL sweep (WT103 / PTB / LAMBADA / PG19 / arXiv).
	$(PY) -m eval.zero_shot --checkpoint $(CKPT_FEAS) \
		--n-samples 4 --device cuda --use-ema \
		--json-out report/zero_shot/feasibility.json \
		--markdown-out report/zero_shot/feasibility.md

# ---------------------------------------------------------------------------
# Quality / efficiency eval suite. Assumes `make gen` (or equivalent) has
# produced a SampleRecord JSONL at $(SAMPLES_JSONL). Override to point at a
# different run's samples.
# ---------------------------------------------------------------------------
SAMPLES_JSONL ?= report/samples/smoke.jsonl

diversity:  ## Self-BLEU + distinct-n + repetition on $(SAMPLES_JSONL).
	$(PY) -m eval.diversity --samples $(SAMPLES_JSONL) \
		--json-out report/diversity/$(notdir $(basename $(SAMPLES_JSONL))).json

gen-ppl:  ## Generation PPL under frozen GPT-2 on $(SAMPLES_JSONL).
	$(PY) -m eval.gen_ppl --samples $(SAMPLES_JSONL) \
		--scorer gpt2 --device cuda \
		--json-out report/gen_ppl/$(notdir $(basename $(SAMPLES_JSONL))).json

mauve:  ## MAUVE vs WT103 validation on $(SAMPLES_JSONL).
	$(PY) -m eval.mauve --samples $(SAMPLES_JSONL) \
		--refs-hf "wikitext|wikitext-103-raw-v1|validation|text" \
		--featurize-model gpt2 --device cuda \
		--json-out report/mauve/$(notdir $(basename $(SAMPLES_JSONL))).json

quality:  ## Diversity + Gen-PPL + MAUVE all in one (MAUVE refs: WT103-val).
	$(PY) -m eval.quality --samples $(SAMPLES_JSONL) \
		--device cuda --gen-ppl-scorer gpt2 \
		--mauve-refs-hf "wikitext|wikitext-103-raw-v1|validation|text" \
		--json-out report/quality/$(notdir $(basename $(SAMPLES_JSONL))).json

efficiency:  ## Sampler efficiency sweep (NFE / wall-clock / peak memory).
	$(PY) -m eval.efficiency --checkpoint $(CKPT_FEAS) --device cuda \
		--configs "adaptive=off,on ; block_size=4 ; batch_size=4,16 ; num_steps=10,20" \
		--repeat 3 --warmup 1 \
		--jsonl-out report/efficiency/feasibility.jsonl \
		--markdown-out report/efficiency/feasibility.md

collect-results:  ## Collect runs from wandb into CSV/markdown/LaTeX tables.
	$(PY) -m eval.collect_results --project abd3 \
		--group-by config.algo.name \
		--csv-out report/tables/all_runs.csv \
		--markdown-out report/tables/summary.md \
		--latex-out report/tables/summary.tex \
		--latex-caption "ABD3 ablation results" \
		--latex-label "tab:abd3-ablation"

lint:  ## Lint + format check (no modifications). What CI runs.
	ruff check abd3/ eval/ tools/ tests/
	ruff format --check abd3/ eval/ tools/ tests/

format:  ## Auto-format and auto-fix with ruff.
	ruff check --fix abd3/ eval/ tools/ tests/
	ruff format abd3/ eval/ tools/ tests/

check: lint test  ## Lint + tests. What CI runs.

clean:  ## Remove caches and editor leftovers.
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) \
		-not -path './.venv/*' -exec rm -rf {} +
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete
