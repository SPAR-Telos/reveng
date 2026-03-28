# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**reveng** is a research project for measuring goal-directedness in AI agents using custom MiniGrid environments.

## Commands

All commands use `uv` as the package manager.

```bash
# Setup
uv sync                    # Install dependencies from lock file
source .venv/bin/activate  # Activate virtual environment

# Testing
make test                  # Run all tests
uvx pytest -n auto -vv    # Run tests with parallelism
uvx pytest tests/test_goal_exists.py -vv  # Run a single test file

# Linting / Formatting
make check-style           # Check without fixing (ruff check + format --check)
make fix-style             # Auto-fix issues

# CLI
reveng-cli --help          # Main entry point (after install)
```

**Ruff configuration:** line length 119, Google-style docstrings.

## Architecture

### Data Flow

1. **Environments** (`environment_generator/`) — Custom MiniGrid environments (`Simple2DNavigationEnv`, coin/key/rooms variants) with configurable size, maze complexity, and observation modes (fog of war, full observability via wrappers in `wrappers/`).

2. **Agents** (`agents/`) — Abstract `Agent` base class. Implementations: `LLMAgent` (uses LLM to select actions), `RandomAgent`, `AlphaStarAgent`. `LLMAgent` extends both `Agent` and `BaseLLMInterface`.

3. **LLM Interface** (`llm_interface.py`) — `BaseLLMInterface` wraps litellm with Jinja2 template rendering (`templates/`), retry logic (tenacity), cost tracking, and Pydantic response format validation. Special handling for Qwen extended thinking.

4. **Trajectory Generation** (`commands/`, `trajectory_generator/`) — CLI via `reveng-cli` (tyro-based) orchestrates batch trajectory collection with rate limiting and parallelization. Supports upload to HuggingFace.

5. **Scoring** (`scoring/`) — `trajectory_scorer.py` and `preference_elicitation.py` evaluate trajectories. Uses LLM-based preference elicitation with Jinja2 templates.

6. **Analysis** (`analysis/`, `analysis_decoded_grids/`, `experiments/`) — Post-processing pipelines for evaluating agent goal-directedness.

### Key Types (`datatypes.py`)

- `Step`: Single env step (observation, action, reward, agent position)
- `Trajectory`: Sequence of `Step`s with final reward and metadata
- `Action`: Enum LEFT/RIGHT/UP/DOWN (values 0–3) with JSON serialization
- `PreferenceResult`, `ComparisonResult`, `PreferenceAnalysis`: Preference elicitation structures

### CLI Entry Points (`commands/cli.py`)

Subcommands: `get_trajectory`, `get_trajectories`, `get_trajectories_multiple_per_grid`, `get_trajectory_key_door_env`, `get_trajectories_key_door_env`, `upload_trajectories_dir`.

### Fixed-Grid Multi-Trajectory Script (`scripts/get_trajectories_fixed_key_door_grid.py`)

Standalone script that generates N trajectories on a **single fixed key-door grid** with varied agent start positions. Designed for IRL / cost-function learning where the MDP must be held constant.

```bash
python scripts/get_trajectories_fixed_key_door_grid.py \
  --num-trajectories 100 \
  --rooms-per-side 2 \
  --seed 42 \
  --max-workers 10 \
  --output-dir ~/Desktop/fixed_key_door_grid
```

**How it works:**
- Creates `RoomsMinigridEnv` and calls `reset()` once to fix the grid (walls, key, door, goal)
- For each trajectory, `deepcopy(master_env)` preserves the full grid state (key/door are removed from grid during gameplay, so `safe_reset()` alone is insufficient)
- Agent start position is randomized per trajectory (sampled from all valid empty cells) for state space coverage
- Uses `use_safe_reset=True` + `skip_reset=True` in `generate_trajectory()` to prevent grid regeneration on the deepcopy
- `RoomsMinigridEnv` overrides `_gen_grid()` without setting `_initial_agent_pos`, so the script sets it manually after `reset()` for `safe_reset()` compatibility
- Parallelized via `ThreadPoolExecutor`, one fresh `LLMAgent` per thread

**Outputs per trajectory:** JSON with `grid_params` (including `traj_id`), `model_params`, `prompt` (token-level), `steps` (per-step `grid_state`, `carrying_key`, `agent_action`, `output_tokens` with logprobs). Plus a shared `grid_layout.json`.

**API key:** Requires `TOGETHER_API_KEY` in `/Users/wws/reveng/.env` (loaded via `BaseLLMInterface`).
