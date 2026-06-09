.PHONY: demo-cast install test lint help mini mini-mode mini-local-mode mini-r1 mini-simple mini-aug mini-mixed mini-aug-local mini-mixed-local full e v local

UV := UV_CACHE_DIR=/i/e/.uv-cache uv
MINI_DATASET := hf://aicrowd/arc-whestbench-public-2026
MINI_MLPS ?= 5
BUDGET ?= 68000000000
WALL_TIME ?= 60

mini:
	$(UV) run whest run --estimator estimator.py \
   --dataset $(MINI_DATASET) \
   --split mini --runner subprocess --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-mode:
	WHEST_K3_MODE=$(MODE) $(UV) run whest run --estimator estimator.py \
   --dataset $(MINI_DATASET) \
   --split mini --runner subprocess --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-local-mode:
	WHEST_K3_MODE=$(MODE) $(UV) run whest run --estimator estimator.py \
   --dataset $(MINI_DATASET) \
   --split mini --runner local --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-r1:
	$(MAKE) mini-mode MODE=r1 BUDGET=$(BUDGET) MINI_MLPS=$(MINI_MLPS) WALL_TIME=$(WALL_TIME)

mini-simple:
	$(MAKE) mini-mode MODE=none BUDGET=$(BUDGET) MINI_MLPS=$(MINI_MLPS) WALL_TIME=$(WALL_TIME)

mini-aug:
	$(MAKE) mini-mode MODE=r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

mini-mixed:
	$(MAKE) mini-mode MODE=last4_r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

mini-aug-local:
	$(MAKE) mini-local-mode MODE=r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

mini-mixed-local:
	$(MAKE) mini-local-mode MODE=last4_r1_slices_k211 BUDGET=1000000000000000 MINI_MLPS=$(MINI_MLPS) WALL_TIME=240

full:
	$(UV) run whest run --estimator estimator.py \
   --dataset $(MINI_DATASET) \
   --split mini --runner subprocess --fail-fast

e:
	$(UV) run python estimator.py

v:
	$(UV) run whest validate --estimator estimator.py


local:
	$(UV) run whest run --estimator estimator.py \
   --dataset $(MINI_DATASET) \
   --split mini --runner local

s:
	uv run whest package --estimator estimator.py --output submission.tar.gz

help:
	@echo "Targets:"
	@echo "  install     uv sync --group dev"
	@echo "  test        run pytest"
	@echo "  lint        run ruff check"
	@echo "  demo-cast   record asciinema cast for README opener"

install:
	$(UV) sync --group dev

test:
	$(UV) run pytest tests/ -v

lint:
	$(UV) run ruff check .

# Records a fresh-clone walkthrough cast and renders to GIF.
# Sequence: git clone -> cd -> uv sync -> python estimator.py -> whest validate
# Output: assets/demo.cast (committed) + assets/demo.gif (committed).
# Requires: asciinema and agg (brew install asciinema agg).
demo-cast:
	@command -v asciinema >/dev/null || (echo "Install asciinema first: brew install asciinema"; exit 1)
	@command -v agg >/dev/null || (echo "Install agg first: brew install agg"; exit 1)
	@DEMO_DIR=$$(mktemp -d) && \
	  cp scripts/record-demo.sh "$$DEMO_DIR/demo.sh" && \
	  chmod +x "$$DEMO_DIR/demo.sh" && \
	  TERM=xterm-256color FORCE_COLOR=1 asciinema rec --overwrite --idle-time-limit=2 \
	    --window-size 90x36 \
	    --title="whest-starterkit: clone, sync, estimate, validate" \
	    --command="bash $$DEMO_DIR/demo.sh $$DEMO_DIR" assets/demo.cast && \
	  rm -rf "$$DEMO_DIR"
	agg --theme monokai --font-size 14 --last-frame-duration 5 \
	  --cols 90 --rows 36 assets/demo.cast assets/demo.gif
