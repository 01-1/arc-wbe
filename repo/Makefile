.PHONY: demo-cast demo-cast-headless install test lint help shadow-snapshot shadow-watch shadow-status random-mini-dataset mini mini-subprocess mini-mode mini-local-mode mini-r1 mini-simple mini-aug mini-mixed mini-aug-local mini-mixed-local compare-residual-pairs modal-small fly fly-mode fly-large fly-large-build fly-large-fast fly-large-dry fly-truth fly-truth-dry truth-bank full e v local

HF_HOME ?= /i/e/.cache/huggingface
HF_DATASETS_CACHE ?= $(HF_HOME)/datasets
UV := HF_HOME=$(HF_HOME) HF_DATASETS_CACHE=$(HF_DATASETS_CACHE) UV_CACHE_DIR=/i/e/.uv-cache uv
SHADOW_HISTORY_DIR ?= .shadow-history
SHADOW_INTERVAL_SECONDS ?= 5
SHADOW_QUIET_SECONDS ?= 60
SHADOW_MIN_COMMIT_SECONDS ?= 120
FLOPSCOPE_GPU ?= 1
MINI_SOURCE_DATASET ?= hf://aicrowd/arc-whestbench-public-2026
MINI_SOURCE_SPLIT ?= mini
MINI_RANDOM_SEED ?= 20260627
MINI_MLPS ?= 5
RANDOM_MINI_DATASET ?= .cache/whestbench/random-mini-n$(MINI_MLPS)-seed$(MINI_RANDOM_SEED)
RANDOM_MINI_METADATA := $(RANDOM_MINI_DATASET)/metadata.json
PAIRS ?= 100
BUDGET ?= 272000000000
WALL_TIME ?= 60
RESIDUAL_WALL_TIME_MULTIPLIER ?= 2.0
MINI_RESIDUAL_WALL_TIME_MULTIPLIER ?= 0.232
OLD_ESTIMATOR ?= /tmp/estimator_old.py
NEW_ESTIMATOR ?= estimator.py
WHEST := FLOPSCOPE_GPU=$(FLOPSCOPE_GPU) $(UV) run python scripts/whest_with_residual_multiplier.py --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER) --
MINI_WHEST := FLOPSCOPE_GPU=$(FLOPSCOPE_GPU) $(UV) run python scripts/whest_with_residual_multiplier.py --residual-wall-time-multiplier $(MINI_RESIDUAL_WALL_TIME_MULTIPLIER) --
MIN_CALL_MS ?= 0
MODAL_MLPS ?= 3
MODAL_CORES ?= 32
MODAL_MEMORY ?= 65536
MODAL_REGION ?= us-east-1
FLY_APP ?= whest-timing-20260627
FLY_MLPS ?= 100
FLY_REGION ?= ewr
FLY_VM_SIZE ?= shared-cpu-8x
FLY_LAUNCH_CONCURRENCY ?= 100
FLY_MIN_RESULTS ?= 80
FLY_MAX_RESULT_SECONDS ?= 45
FLY_FAST_TIMEOUT ?= 70s
FLY_AWS_PROFILE ?= whest-tigris
FLY_OBJECT_BUCKET ?= whest-fly-runner-20260628
FLY_OBJECT_PREFIX ?= whest/fly-mlps
FLY_DATASET_FINGERPRINT ?= 50c6efdca4059f0e
FLY_DATASET_URL_CACHE ?= .cache/whestbench/fly-dataset-urls-$(FLY_OBJECT_BUCKET)-$(FLY_DATASET_FINGERPRINT)-n$(FLY_MLPS).json
FLY_DATASET_FLAGS ?= --dataset-fingerprint $(FLY_DATASET_FINGERPRINT) --dataset-url-cache $(FLY_DATASET_URL_CACHE)
FLY_OBJECT_ENDPOINT_URL ?=
FLY_OBJECT_FLAGS ?= --object-bucket $(FLY_OBJECT_BUCKET) --object-base-url https://whest-fly-runner-20260628.fly.storage.tigris.dev
FLY_ESTIMATOR ?= estimator.py
FLY_ESTIMATOR_OBJECT_FLAGS ?= --estimator-object-bucket whest-fly-runner-20260628 --estimator-object-base-url https://whest-fly-runner-20260628.fly.storage.tigris.dev
FLY_ESTIMATOR_URL_CACHE ?= .cache/whestbench/fly-estimator-url-$(FLY_OBJECT_BUCKET).json
FLY_ESTIMATOR_FLAGS ?= --estimator-url-cache $(FLY_ESTIMATOR_URL_CACHE)
FLY_IMAGE_LABEL ?= whest-runner-local-scale-20260628
FLY_BUILD_FLAGS ?=
FLY_WORKER_RUNNER ?= local
FLY_RESIDUAL_COMPUTE_SCALE ?= 0.1
FLY_RUN_FLAGS ?= --launch-method api --detach-launch --ignore-destroy-time --summary-only --quiet-commands --no-timing-summary --format json --worker-runner $(FLY_WORKER_RUNNER) --presign-urls --presign-expires-in 604800 --linger-after-result 0 --min-results $(FLY_MIN_RESULTS) --max-result-seconds $(FLY_MAX_RESULT_SECONDS) --residual-compute-scale $(FLY_RESIDUAL_COMPUTE_SCALE)
FLY_UPLOAD_FLAGS ?=
FLY_TRUTH_SCRIPT ?= scripts/fly_truth_entrypoint.py
FLY_TRUTH_JSONL ?= analysis/truth_bank/raw_results.jsonl
FLY_TRUTH_TARGET_SECONDS ?= 60
FLY_TRUTH_MAX_RESULT_SECONDS ?= 420
FLY_TRUTH_CHUNK_PAIRS ?= 1024
FLY_TRUTH_MIN_PAIRS ?= 1024
FLY_TRUTH_SEED_LABEL ?= arc-whest-fly-truth-bank-20260706-v1
FLY_TRUTH_FLAGS ?= --task truth --research-script $(FLY_TRUTH_SCRIPT) --no-upload-estimator --result-jsonl $(FLY_TRUTH_JSONL) --truth-target-seconds $(FLY_TRUTH_TARGET_SECONDS) --truth-chunk-pairs $(FLY_TRUTH_CHUNK_PAIRS) --truth-min-pairs $(FLY_TRUTH_MIN_PAIRS) --truth-seed-label $(FLY_TRUTH_SEED_LABEL) --min-results $(FLY_MLPS) --max-result-seconds $(FLY_TRUTH_MAX_RESULT_SECONDS) --result-wait-timeout $(FLY_TRUTH_MAX_RESULT_SECONDS)

random-mini-dataset: $(RANDOM_MINI_METADATA)

shadow-snapshot:
	$(UV) run python scripts/shadow_history.py --shadow-dir $(SHADOW_HISTORY_DIR) snapshot

shadow-watch:
	$(UV) run python scripts/shadow_history.py --shadow-dir $(SHADOW_HISTORY_DIR) watch \
   --interval-seconds $(SHADOW_INTERVAL_SECONDS) \
   --quiet-seconds $(SHADOW_QUIET_SECONDS) \
   --min-commit-seconds $(SHADOW_MIN_COMMIT_SECONDS)

shadow-status:
	$(UV) run python scripts/shadow_history.py --shadow-dir $(SHADOW_HISTORY_DIR) status

$(RANDOM_MINI_METADATA):
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
	$(MINI_WHEST) run --estimator estimator.py \
   --dataset $(RANDOM_MINI_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner local --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-subprocess: random-mini-dataset
	$(MINI_WHEST) run --estimator estimator.py \
   --dataset $(RANDOM_MINI_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner subprocess --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-mode: random-mini-dataset
	WHEST_K3_MODE=$(MODE) $(MINI_WHEST) run --estimator estimator.py \
   --dataset $(RANDOM_MINI_DATASET) \
   --split $(MINI_SOURCE_SPLIT) --runner subprocess --fail-fast --n-mlps=$(MINI_MLPS) --flop-budget $(BUDGET) \
   --wall-time-limit $(WALL_TIME)

mini-local-mode: random-mini-dataset
	WHEST_K3_MODE=$(MODE) $(MINI_WHEST) run --estimator estimator.py \
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

modal-small:
	modal run cloud/modal_runner.py \
   --n-mlps $(MODAL_MLPS) --seed $(MINI_RANDOM_SEED) \
   --cores $(MODAL_CORES) --memory $(MODAL_MEMORY) --region $(MODAL_REGION) \
   --flop-budget $(BUDGET) --wall-time-limit $(WALL_TIME) \
   --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER)

fly-large:
	@test -n "$(FLY_APP)" || (echo "Set FLY_APP=<app-name>"; exit 1)
	AWS_PROFILE=$(FLY_AWS_PROFILE) $(UV) run python cloud/fly_runner.py \
   --app $(FLY_APP) --n-mlps $(FLY_MLPS) --seed $(MINI_RANDOM_SEED) \
   --region $(FLY_REGION) --vm-size $(FLY_VM_SIZE) \
   --launch-concurrency $(FLY_LAUNCH_CONCURRENCY) \
   --object-prefix $(FLY_OBJECT_PREFIX) $(FLY_DATASET_FLAGS) $(FLY_OBJECT_FLAGS) \
   --estimator $(FLY_ESTIMATOR) --image-label $(FLY_IMAGE_LABEL) \
   $(FLY_ESTIMATOR_OBJECT_FLAGS) $(FLY_ESTIMATOR_FLAGS) \
   --flop-budget $(BUDGET) --wall-time-limit $(WALL_TIME) \
   --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER) \
   $(FLY_BUILD_FLAGS) $(FLY_RUN_FLAGS)

fly-large-build:
	@test -n "$(FLY_APP)" || (echo "Set FLY_APP=<app-name>"; exit 1)
	AWS_PROFILE=$(FLY_AWS_PROFILE) $(UV) run python cloud/fly_runner.py \
   --app $(FLY_APP) --n-mlps $(FLY_MLPS) --seed $(MINI_RANDOM_SEED) \
   --region $(FLY_REGION) --vm-size $(FLY_VM_SIZE) \
   --launch-concurrency $(FLY_LAUNCH_CONCURRENCY) \
   --object-prefix $(FLY_OBJECT_PREFIX) $(FLY_OBJECT_FLAGS) \
   --estimator $(FLY_ESTIMATOR) --image-label $(FLY_IMAGE_LABEL) \
   $(FLY_ESTIMATOR_OBJECT_FLAGS) $(FLY_ESTIMATOR_FLAGS) \
   --flop-budget $(BUDGET) --wall-time-limit $(WALL_TIME) \
   --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER) \
   --build-only $(FLY_UPLOAD_FLAGS) $(FLY_BUILD_FLAGS)

fly-large-fast:
	@test -n "$(FLY_APP)" || (echo "Set FLY_APP=<app-name>"; exit 1)
	@timeout --foreground $(FLY_FAST_TIMEOUT) env PYTHONUNBUFFERED=1 AWS_PROFILE=$(FLY_AWS_PROFILE) $(UV) run python cloud/fly_runner.py \
   --app $(FLY_APP) --n-mlps $(FLY_MLPS) --seed $(MINI_RANDOM_SEED) \
   --region $(FLY_REGION) --vm-size $(FLY_VM_SIZE) \
   --launch-concurrency $(FLY_LAUNCH_CONCURRENCY) \
   --object-prefix $(FLY_OBJECT_PREFIX) $(FLY_DATASET_FLAGS) $(FLY_OBJECT_FLAGS) \
   --estimator $(FLY_ESTIMATOR) --image-label $(FLY_IMAGE_LABEL) \
   $(FLY_ESTIMATOR_OBJECT_FLAGS) $(FLY_ESTIMATOR_FLAGS) \
   --flop-budget $(BUDGET) --wall-time-limit $(WALL_TIME) \
   --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER) \
   --skip-build $(FLY_RUN_FLAGS)

fly: fly-large-fast

fly-mode:
	@test -n "$(MODE)" || (echo "Set MODE=<mode>"; exit 1)
	$(MAKE) fly-large-fast FLY_RUN_FLAGS="$(FLY_RUN_FLAGS) --mode $(MODE)"

fly-large-dry:
	@test -n "$(FLY_APP)" || (echo "Set FLY_APP=<app-name>"; exit 1)
	AWS_PROFILE=$(FLY_AWS_PROFILE) $(UV) run python cloud/fly_runner.py \
   --app $(FLY_APP) --n-mlps $(FLY_MLPS) --seed $(MINI_RANDOM_SEED) \
   --region $(FLY_REGION) --vm-size $(FLY_VM_SIZE) \
   --launch-concurrency $(FLY_LAUNCH_CONCURRENCY) \
   --object-prefix $(FLY_OBJECT_PREFIX) $(FLY_DATASET_FLAGS) $(FLY_OBJECT_FLAGS) \
   --estimator $(FLY_ESTIMATOR) --image-label $(FLY_IMAGE_LABEL) \
   $(FLY_ESTIMATOR_OBJECT_FLAGS) $(FLY_ESTIMATOR_FLAGS) \
   --flop-budget $(BUDGET) --wall-time-limit $(WALL_TIME) \
   --residual-wall-time-multiplier $(RESIDUAL_WALL_TIME_MULTIPLIER) \
   --dry-run

fly-truth:
	@test -n "$(FLY_APP)" || (echo "Set FLY_APP=<app-name>"; exit 1)
	AWS_PROFILE=$(FLY_AWS_PROFILE) $(UV) run python cloud/fly_runner.py \
   --app $(FLY_APP) --n-mlps $(FLY_MLPS) \
   --region $(FLY_REGION) --vm-size $(FLY_VM_SIZE) \
   --launch-concurrency $(FLY_LAUNCH_CONCURRENCY) \
   --object-bucket $(FLY_OBJECT_BUCKET) --object-base-url https://whest-fly-runner-20260628.fly.storage.tigris.dev \
   --image-label $(FLY_IMAGE_LABEL) \
   $(FLY_RUN_FLAGS) \
   $(FLY_TRUTH_FLAGS)

fly-truth-dry:
	@test -n "$(FLY_APP)" || (echo "Set FLY_APP=<app-name>"; exit 1)
	AWS_PROFILE=$(FLY_AWS_PROFILE) $(UV) run python cloud/fly_runner.py \
   --app $(FLY_APP) --n-mlps $(FLY_MLPS) \
   --region $(FLY_REGION) --vm-size $(FLY_VM_SIZE) \
   --launch-concurrency $(FLY_LAUNCH_CONCURRENCY) \
   --object-bucket $(FLY_OBJECT_BUCKET) --object-base-url https://whest-fly-runner-20260628.fly.storage.tigris.dev \
   --image-label $(FLY_IMAGE_LABEL) \
   $(FLY_RUN_FLAGS) \
   $(FLY_TRUTH_FLAGS) --dry-run

truth-bank:
	$(UV) run python scripts/assemble_truth_bank.py \
   --input-jsonl $(FLY_TRUTH_JSONL) \
   --output-dir analysis/truth_bank \
   --expected-count $(FLY_MLPS)

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
	@echo "  shadow-snapshot         commit a private .gitignore-respecting repo snapshot"
	@echo "  shadow-watch            keep private repo-scale shadow history running"
	@echo "  shadow-status           show recent shadow-history snapshots"
	@echo "  mini                    fast local 3-MLP smoke check"
	@echo "  mini-subprocess         isolated 3-MLP subprocess check"
	@echo "  mini-mode               cached mini split with MODE=<mode>"
	@echo "  compare-residual-pairs  paired old/new residual benchmark"
	@echo "  modal-small             run small HF-derived batches on Modal with many CPU cores"
	@echo "  fly-large               build image and run Fly.io EWR Machines using existing object-store archives"
	@echo "  fly-large-build         upload archives and prebuild/push the small Fly code image"
	@echo "  fly                     short alias for fly-large-fast"
	@echo "  fly-mode                run Fly with MODE=<mode> through the fast path"
	@echo "  fly-large-fast          run Fly Machines from the prebuilt image and existing object-store archives"
	@echo "  fly-large-dry           print Fly commands without running them"
	@echo "  fly-truth               generate fresh-seed Fly truth-bank raw JSONL"
	@echo "  fly-truth-dry           print Fly truth-bank commands without running them"
	@echo "  truth-bank              assemble analysis/truth_bank from Fly truth JSONL"
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
