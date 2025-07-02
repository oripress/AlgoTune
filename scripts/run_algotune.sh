#!/usr/bin/env bash
set -euo pipefail

# run_algobench.sh - Bash entry point for AlgoTune
# This script handles timing evaluations and test suite execution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
TARGET_TIME_MS=100
STANDALONE=false
SEQUENTIAL=false
# Helper runs on host by default (use --helper-singularity to override)
HELPER_SINGULARITY=false
TASK=""
TASK_LIST=""
TASK_LIST_FILE=""
DATA_DIR=""
TEST_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target-time-ms)
            TARGET_TIME_MS="$2"
            shift 2
            ;;
        --standalone)
            STANDALONE=true
            shift
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --helper-singularity)
            HELPER_SINGULARITY=true
            shift
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --task-list)
            TASK_LIST="$2"
            shift 2
            ;;
        --task-list-file)
            TASK_LIST_FILE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "AlgoTune - Run timing evaluations or test suite"
            echo ""
            echo "Options:"
            echo "  --target-time-ms NUM    Target time in milliseconds (default: 100)"
            echo "  --standalone           Force standalone mode (no SLURM)"
            echo "  --sequential           Process tasks sequentially" 
            echo "  --helper-singularity   Run submission helper inside Singularity (optional)"
            echo "  --task NAME            Single task to run"
            echo "  --task-list NAME       Predefined task list"
            echo "  --task-list-file FILE  Custom task list file"
            echo "  --data-dir DIR         Data directory override"
            echo "  --test                 Run test suite with dummy LLM"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --target-time-ms 100        # Run timing evaluations"
            echo "  $0 --test                     # Run test suite"
            echo "  $0 --test --standalone        # Run tests in standalone mode"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure Python can import in-repo packages regardless of CWD
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# After parsing args block, before auto-detect SLURM add export when test mode
if [ "$TEST_MODE" = true ]; then
    export TIMING_OVERHEAD_DEBUG=1
    echo "⏱️  TIMING_OVERHEAD_DEBUG enabled (env var exported)"
fi

# Auto-detect SLURM availability
if command -v squeue &> /dev/null && command -v sbatch &> /dev/null && [ "$STANDALONE" = false ]; then
    if [ "$TEST_MODE" = true ]; then
        echo "🤖 SLURM detected - using SLURM mode for tests"
    else
        echo "🤖 SLURM detected - using SLURM mode"
    fi
    SLURM_MODE=true
else
    if [ "$TEST_MODE" = true ]; then
        echo "💻 Using standalone mode for tests"
    else
        echo "💻 Using standalone mode"
    fi
    SLURM_MODE=false
    STANDALONE=true
fi

# Handle test mode
if [ "$TEST_MODE" = true ]; then
    # Check if test input directory exists
    TESTS_INPUT_DIR="$PROJECT_ROOT/AlgoTuner/tests/inputs"
    if [ ! -d "$TESTS_INPUT_DIR" ]; then
        echo "❌ Tests input directory not found: $TESTS_INPUT_DIR"
        exit 1
    fi

    # Count test files
    TEST_COUNT=$(find "$TESTS_INPUT_DIR" -name "*.txt" | wc -l)
    if [ "$TEST_COUNT" -eq 0 ]; then
        echo "❌ No test files found in $TESTS_INPUT_DIR"
        exit 1
    fi

    echo "📋 Found $TEST_COUNT test files"

    # Build test command arguments
    TEST_ARGS=()
    if [ "$STANDALONE" = true ]; then
        TEST_ARGS+=(--standalone)
    fi

    # Check if Python dependencies are available for tests
    echo "🔍 Checking Python dependencies..."

    if [ "$SLURM_MODE" = true ]; then
        # SLURM mode - submit test jobs
        echo "🤖 SLURM mode - submitting test jobs to queue"
        python3 "$PROJECT_ROOT/algotune.py" test "${TEST_ARGS[@]}"
    else
        # Standalone mode - need full dependencies
        if python3 -c "import orjson, numpy" 2>/dev/null; then
            echo "✅ Dependencies available - running tests directly"
            python3 "$PROJECT_ROOT/algotune.py" test "${TEST_ARGS[@]}"
        else
            echo "📦 Dependencies missing - using Singularity for tests..."
            
            # Load configuration for Singularity image (try unified config first)
            if [ -f "$PROJECT_ROOT/config.env" ]; then
                echo "📋 Loading unified config from config.env"
                source "$PROJECT_ROOT/config.env"
            elif [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
                echo "📋 Loading config from slurm/run_config.env (consider migrating to config.env)"
                source "$PROJECT_ROOT/slurm/run_config.env"
            else
                echo "❌ No configuration file found"
                echo "Please create config.env or slurm/run_config.env"
                exit 1
            fi
            
            if [ -n "${SINGULARITY_IMAGE:-}" ] && [ -f "$SINGULARITY_IMAGE" ]; then
                echo "🐍 Using Singularity: $SINGULARITY_IMAGE"
                
                # Build bind mounts
                BIND_MOUNTS=("$PROJECT_ROOT:/app")
                
                if [ -n "${DATA_DIR:-}" ] && [ "$DATA_DIR" != "$PROJECT_ROOT" ]; then
                    mkdir -p "$DATA_DIR" || echo "Warning: Could not create DATA_DIR $DATA_DIR"
                    BIND_MOUNTS+=("$DATA_DIR:$DATA_DIR")
                fi
                
                if [ -n "${TEMP_DIR_STORAGE:-}" ] && [ "$TEMP_DIR_STORAGE" != "/tmp" ] && [ "$TEMP_DIR_STORAGE" != "$PROJECT_ROOT" ]; then
                    mkdir -p "$TEMP_DIR_STORAGE" || echo "Warning: Could not create TEMP_DIR $TEMP_DIR_STORAGE"
                    BIND_MOUNTS+=("$TEMP_DIR_STORAGE:$TEMP_DIR_STORAGE")
                fi
                
                # Bind mount CODE_DIR if it's different from existing mounts
                if [ -n "${CODE_DIR:-}" ] && [ "$CODE_DIR" != "$PROJECT_ROOT" ] && [ "$CODE_DIR" != "${DATA_DIR:-}" ] && [ "$CODE_DIR" != "${TEMP_DIR_STORAGE:-}" ]; then
                    mkdir -p "$CODE_DIR" || echo "Warning: Could not create CODE_DIR $CODE_DIR"
                    BIND_MOUNTS+=("$CODE_DIR:$CODE_DIR")
                fi
                
                # Build singularity command for tests
                SING_CMD=(singularity exec)
                for bind in "${BIND_MOUNTS[@]}"; do
                    SING_CMD+=(--bind "$bind")
                done
                
                SING_CMD+=(
                    --env "PYTHONPATH=/app"
                    --env "CODE_DIR=/app"
                    --env "AGENT_MODE=1"
                    --env "DATA_DIR=${DATA_DIR:-/app/data}"
                    --env "TEMP_DIR_STORAGE=${TEMP_DIR_STORAGE:-/tmp}"
                    "$SINGULARITY_IMAGE"
                    python3 /app/algotune.py test
                )
                
                echo "🔗 Bind mounts: ${BIND_MOUNTS[*]}"
                "${SING_CMD[@]}" "${TEST_ARGS[@]}"
            else
                echo "❌ Singularity image not found: ${SINGULARITY_IMAGE:-<not set>}"
                echo "Please either:"
                echo "1. Install dependencies: pip install -e ."
                echo "2. Configure SINGULARITY_IMAGE in config.env or slurm/run_config.env"
                exit 1
            fi
        fi
    fi

    echo "🏁 Test execution completed"
    exit 0
fi

# Original timing evaluation logic below...

# Build Python command arguments
PYTHON_ARGS=(--target-time-ms "$TARGET_TIME_MS")

if [ "$STANDALONE" = true ]; then
    PYTHON_ARGS+=(--standalone)
fi

if [ "$SEQUENTIAL" = true ]; then
    PYTHON_ARGS+=(--sequential)
fi

if [ -n "$TASK" ]; then
    PYTHON_ARGS+=(--task "$TASK")
fi

if [ -n "$TASK_LIST" ]; then
    PYTHON_ARGS+=(--task-list "$TASK_LIST")
fi

if [ -n "$TASK_LIST_FILE" ]; then
    PYTHON_ARGS+=(--task-list-file "$TASK_LIST_FILE")
fi

if [ -n "$DATA_DIR" ]; then
    PYTHON_ARGS+=(--data-dir "$DATA_DIR")
fi

# Create temporary CODE_DIR for auxiliary files if not set
if [ -z "${CODE_DIR:-}" ]; then
    # Generate unique ID for this run
    UNIQUE_ID=$(python3 -c "import uuid; print(str(uuid.uuid4())[:8])")
    TEMP_CODE_DIR=$(mktemp -d -t "algotune_timing_${UNIQUE_ID}_XXXXXX")
    export CODE_DIR="$TEMP_CODE_DIR"
    echo "📁 Created temporary CODE_DIR for auxiliary files: $CODE_DIR"
    # Set cleanup trap
    trap 'rm -rf "$TEMP_CODE_DIR" 2>/dev/null || true' EXIT
else
    echo "📁 Using existing CODE_DIR: $CODE_DIR"
fi

# Check if Python dependencies are available
echo "🔍 Checking Python dependencies..."

if [ "$SLURM_MODE" = true ]; then
    if [ "$HELPER_SINGULARITY" = true ]; then
        echo "🐍 Running submission helper inside Singularity (SLURM mode)"

        # Load configuration for Singularity image
        if [ -f "$PROJECT_ROOT/config.env" ]; then
            source "$PROJECT_ROOT/config.env"
        elif [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
            source "$PROJECT_ROOT/slurm/run_config.env"
        else
            echo "❌ No configuration file found (config.env or slurm/run_config.env)"
            exit 1
        fi

        if [ -z "${SINGULARITY_IMAGE:-}" ] || [ ! -f "$SINGULARITY_IMAGE" ]; then
            echo "❌ Singularity image not found: ${SINGULARITY_IMAGE:-<not set>}"
            exit 1
        fi

        # Build bind mounts (minimal set, avoid overriding container binaries)
        BIND_MOUNTS=("$PROJECT_ROOT:/app")
        if [ -n "${DATA_DIR:-}" ] && [ "$DATA_DIR" != "$PROJECT_ROOT" ]; then
            mkdir -p "$DATA_DIR" || true
            BIND_MOUNTS+=("$DATA_DIR:$DATA_DIR")
        fi
        if [ -n "${TEMP_DIR_STORAGE:-}" ] && [ "$TEMP_DIR_STORAGE" != "/tmp" ] && [ "$TEMP_DIR_STORAGE" != "$PROJECT_ROOT" ]; then
            mkdir -p "$TEMP_DIR_STORAGE" || true
            BIND_MOUNTS+=("$TEMP_DIR_STORAGE:$TEMP_DIR_STORAGE")
        fi

        SING_CMD=(singularity exec)
        for bind in "${BIND_MOUNTS[@]}"; do
            SING_CMD+=(--bind "$bind")
        done

        SING_CMD+=(
            --env "PYTHONPATH=/app"
            --env "CODE_DIR=/app"
            --env "DATA_DIR=${DATA_DIR:-/app/data}"
            --env "TEMP_DIR_STORAGE=${TEMP_DIR_STORAGE:-/tmp}"
            "$SINGULARITY_IMAGE"
            python3 /app/scripts/submit_generate_python.py
        )

        echo "🔗 Bind mounts: ${BIND_MOUNTS[*]}"
        "${SING_CMD[@]}" "${PYTHON_ARGS[@]}"
    else
        # Regular host execution
        echo "🤖 SLURM mode - using job submission script"
        python3 "$SCRIPT_DIR/submit_generate_python.py" "${PYTHON_ARGS[@]}"
    fi
else
    # Standalone mode - need full dependencies
    if python3 -c "import orjson, numpy" 2>/dev/null; then
        echo "✅ Dependencies available - running directly"
        python3 "$SCRIPT_DIR/submit_generate_python.py" "${PYTHON_ARGS[@]}"
    else
        echo "📦 Dependencies missing - using Singularity..."
        
        # Load configuration for Singularity image (try unified config first)
        if [ -f "$PROJECT_ROOT/config.env" ]; then
            echo "📋 Loading unified config from config.env"
            source "$PROJECT_ROOT/config.env"
        elif [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
            echo "📋 Loading config from slurm/run_config.env (consider migrating to config.env)"
            source "$PROJECT_ROOT/slurm/run_config.env"
        else
            echo "❌ No configuration file found"
            echo "Please create config.env or slurm/run_config.env"
            exit 1
        fi
        
        if [ -n "${SINGULARITY_IMAGE:-}" ] && [ -f "$SINGULARITY_IMAGE" ]; then
            echo "🐍 Using Singularity: $SINGULARITY_IMAGE"
            
            # Build bind mounts
            BIND_MOUNTS=("$PROJECT_ROOT:/app")
            
            if [ -n "${DATA_DIR:-}" ] && [ "$DATA_DIR" != "$PROJECT_ROOT" ]; then
                mkdir -p "$DATA_DIR" || echo "Warning: Could not create DATA_DIR $DATA_DIR"
                BIND_MOUNTS+=("$DATA_DIR:$DATA_DIR")
            fi
            
            if [ -n "${TEMP_DIR_STORAGE:-}" ] && [ "$TEMP_DIR_STORAGE" != "/tmp" ] && [ "$TEMP_DIR_STORAGE" != "$PROJECT_ROOT" ]; then
                mkdir -p "$TEMP_DIR_STORAGE" || echo "Warning: Could not create TEMP_DIR $TEMP_DIR_STORAGE"
                BIND_MOUNTS+=("$TEMP_DIR_STORAGE:$TEMP_DIR_STORAGE")
            fi
            
            # Bind mount CODE_DIR if it's different from existing mounts
            if [ -n "${CODE_DIR:-}" ] && [ "$CODE_DIR" != "$PROJECT_ROOT" ] && [ "$CODE_DIR" != "${DATA_DIR:-}" ] && [ "$CODE_DIR" != "${TEMP_DIR_STORAGE:-}" ]; then
                mkdir -p "$CODE_DIR" || echo "Warning: Could not create CODE_DIR $CODE_DIR"
                BIND_MOUNTS+=("$CODE_DIR:$CODE_DIR")
            fi
            
            # Build singularity command
            SING_CMD=(singularity exec)
            for bind in "${BIND_MOUNTS[@]}"; do
                SING_CMD+=(--bind "$bind")
            done
            
            SING_CMD+=(
                --env "PYTHONPATH=/app"
                --env "CODE_DIR=/app"
                --env "DATA_DIR=${DATA_DIR:-/app/data}"
                --env "TEMP_DIR_STORAGE=${TEMP_DIR_STORAGE:-/tmp}"
                "$SINGULARITY_IMAGE"
                python3 /app/scripts/submit_generate_python.py
            )
            
            echo "🔗 Bind mounts: ${BIND_MOUNTS[*]}"
            "${SING_CMD[@]}" "${PYTHON_ARGS[@]}"
        else
            echo "❌ Singularity image not found: ${SINGULARITY_IMAGE:-<not set>}"
            echo "Please either:"
            echo "1. Install dependencies: pip install -e ."
            echo "2. Configure SINGULARITY_IMAGE in slurm/run_config.env"
            exit 1
        fi
    fi
fi 