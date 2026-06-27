.PHONY: demo-cast demo-cast-headless install test lint help random-mini-dataset mini mini-mode mini-local-mode mini-r1 mini-simple mini-aug mini-mixed mini-aug-local mini-mixed-local compare-residual-pairs full e v local

HF_HOME ?= /i/e/.cache/huggingface
HF_DATASETS_CACHE ?= $(HF_HOME)/datasets
UV := HF_HOME=$(HF_HOME) HF_DATASETS_CACHE=$(HF_DATASETS_CACHE) UV_CACHE_DIR=/i/e/.uv-cache uv
FLOPSCOPE_GPU ?= 1
MINI_SOURCE_DATASET ?= hf://aicrowd/arc-whestbench-public-2026
MINI_SOURCE_SPLIT ?= mini
MINI_RANDOM_SEED ?= 20260624
MINI_MLPS ?= 3
RANDOM_MINI_DATASET ?= .cache/whestbench/random-mini-n$(MINI_MLPS)-seed$(MINI_RANDOM_SEED)
PAIRS ?= 100
BUDGET ?= 272000000000
WALL_TIME ?= 60
RESIDUAL_WALL_TIME_MULTIPLIER ?= 2.0
OLD_ESTIMATOR ?= /tmp/estimator_old.py
NEW_ESTIMATOR ?= estimator.py
WHEST := FLOPSCOPE_GPU=$(FLOPSCOPE_GPU) $(UV) run python scripts/whest_with_residual_multiplier.py --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER) --
MIN_CALL_MS ?= 0

random-mini-dataset:
	$(UV) run python scripts/randomize_whest_dataset.py \
   --dataset $(MINI_SOURCE_DATASET) --split $(MINI_SOURCE_SPLIT) \
   --n-mlps $(MINI_MLPS) --seed $(MINI_RANDOM_SEED) \
   --output $(RANDOM_MINI_DATASET)

1:
	$(UV) run python scripts/randomize_whest_dataset.py \
   --dataset $(MINI_SOURCE_DATASET) --split $(MINI_SOURCE_SPLIT) \
   --n-mlps 1 --seed $(MINI_RANDOM_SEED) \
   --output .cache/whestbench/random-mini-n1-seed$(MINI_RANDOM_SEED)
	$(WHEST) run --estimator estimator.py \
   --dataset .cache/whestbench/random-mini-n1-seed$(MINI_RANDOM_SEED) \
   --split $(MINI_SOURCE_SPLIT) --runner subprocess --fail-fast --n-mlps=1 --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini: random-mini-dataset
	$(WHEST) run --estimator estimator.py \
   --dataset $(RANDOM_MINI_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner subprocess --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-mode: random-mini-dataset
	WHEST_K3_MODE=$(MODE) $(WHEST) run --estimator estimator.py \
   --dataset $(RANDOM_MINI_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner subprocess --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-local-mode: random-mini-dataset
	WHEST_K3_MODE=$(MODE) $(WHEST) run --estimator estimator.py \
   --dataset $(RANDOM_MINI_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner local --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-r1:
	$(MAKE) mini-mode MODE=r1 BUDGET=$(BUDGET) MINI_MLPS=$(MINI_MLPS) WALL_TIME=$(WALL_TIME)

mini-simple:
	$(MAKE) mini-mode MODE=n1 BUDGET=$(BUDGET) MINI_MLPS=$(MINI_MLPS) WALL_TIME=$(WALL_TIME)

mini-aug:
	$(MAKE) mini-mode MODE=r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

mini-mixed:
	$(MAKE) mini-mode MODE=last4_r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

mini-aug-local:
	$(MAKE) mini-local-mode MODE=r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

mini-mixed-local:
	$(MAKE) mini-local-mode MODE=last4_r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

compare-residual-pairs: random-mini-dataset
	$(UV) run python scripts/compare_residual_pairs.py \
   --old $(OLD_ESTIMATOR) --new $(NEW_ESTIMATOR) \
   --pairs $(PAIRS) --n-mlps $(MINI_MLPS) \
   --dataset $(RANDOM_MINI_DATASET) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME) \
   --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER)

p:
	UV_CACHE_DIR=/i/e/.uv-cache uv run python scripts/profile_residual_tree.py --targeted-min-call-ms $(MIN_CALL_MS)

pu:
	UV_CACHE_DIR=/i/e/.uv-cache uv run python scripts/profile_residual_tree.py --unified-call-tree --targeted-min-call-ms $(MIN_CALL_MS)

full:
	$(WHEST) run --estimator estimator.py \
   --dataset $(MINI_SOURCE_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner subprocess --fail-fast

e:
	FLOPSCOPE_GPU=$(FLOPSCOPE_GPU) $(UV) run python estimator.py

v:
	FLOPSCOPE_GPU=$(FLOPSCOPE_GPU) $(UV) run whest validate --estimator estimator.py


local: random-mini-dataset
	$(WHEST) run --estimator estimator.py \
   --dataset $(RANDOM_MINI_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner local

s:
	uv run whest package --estimator estimator.py --output submission.tar.gz

help:
	@echo "Targets:"
	@echo "  install                 uv sync --group dev"
	@echo "  test                    run pytest"
	@echo "  lint                    run ruff check"
	@echo "  mini                    cached mini split with residual multiplier ($(RESIDUAL_WALL_TIME_MULTIPLIER)x)"
	@echo "  mini-mode               cached mini split with MODE=<mode>"
	@echo "  compare-residual-pairs  paired old/new residual benchmark"
	@echo "  demo-cast               record asciinema cast for README opener (needs a TTY)"
	@echo "  demo-cast-headless      build the cast without a TTY (CI / agent sandboxes)"

install:
	$(UV) sync --group dev

test:
	$(UV) run pytest tests/ -v

lint:
	$(UV) run ruff check .

# Records a fresh-cl1 walkthrough cast and renders to GIF.
# Sequence: git cl1 -> cd -> uv sync -> python estimator.py -> whest validate
# Output: assets/demo.cast (committed) + assets/demo.gif (committed).
# Requires: asciinema and agg (brew install asciinema agg).
demo-cast:
	@command -v asciinema >/dev/null || (echo "Install asciinema first: brew install asciinema"; exit 1)
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	@python3 -c "import os; os.close(os.openpty()[0])" 2>/dev/null || ( \
	  echo "No PTY available (CI / agent sandbox) — 'asciinema rec' needs a terminal."; \
	  echo "Run 'make demo-cast-headless' instead."; exit 1)
	@DEMO_DIR=$$(mktemp -d) && \
	  cp scripts/record-demo.sh "$$DEMO_DIR/demo.sh" && \
	  chmod +x "$$DEMO_DIR/demo.sh" && \
	  TERM=xterm-256color FORCE_COLOR=1 asciinema rec --overwrite --idle-time-limit=2 \
	    --window-size 90x36 \
	    --title="whest-starterkit: cl1, sync, estimate, validate" \
	    --command="bash $$DEMO_DIR/demo.sh $$DEMO_DIR" assets/demo.cast && \
	  rm -rf "$$DEMO_DIR"
	agg --theme monokai --font-size 14 --last-frame-duration 5 \
	  --cols 90 --rows 36 assets/demo.cast assets/demo.gif

# Headless fallback for environments without a PTY (CI / agent sandboxes), where
# 'asciinema rec' can't run. Runs the same sequence, captures real output, strips
# transient spinner frames, assembles assets/demo.cast + renders assets/demo.gif.
# Requires: agg. Keep the sequence in sync with scripts/record-demo.sh.
demo-cast-headless:
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	python3 scripts/record-demo-headless.py
