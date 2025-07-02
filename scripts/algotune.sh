#!/usr/bin/env bash

# AlgoBench launcher
# Unified launcher for both SLURM and standalone operations

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if SLURM is available
HAS_SLURM=false
if command -v sbatch &> /dev/null; then
    HAS_SLURM=true
fi

usage() {
    cat << EOF
AlgoBench Launcher

USAGE:
    $0 <command> [options]

COMMANDS:
    generate [options]                    Generate baseline measurements
    agent [options] <model> [task]...     Run AI agent on tasks
    test [options]                        Run test suite
    list-tasks                            List available tasks
    list-task-lists                       List available task lists
    
OPTIONS:
    --standalone                          Force standalone mode (no SLURM)
    --target-time-ms N                    Target time in milliseconds (for generate)
    --tasks task1,task2                   Specific tasks (for generate)
    
EXAMPLES:
    # Generate baseline data
    $0 generate --target-time-ms 100                    # SLURM if available
    $0 generate --standalone --target-time-ms 100       # Force standalone
    
    # Run agent on specific tasks
    $0 agent o4-mini svm kmeans                          # SLURM if available
    $0 agent --standalone o4-mini svm kmeans             # Force standalone
    
    # Run agent on all tasks
    $0 agent o4-mini                                     # SLURM if available
    
    # Run tests
    $0 test                                              # SLURM if available
    $0 test --standalone                                 # Force standalone
    
    # List available tasks
    $0 list-tasks

For more details, see README.md
EOF
}

# Parse global options
STANDALONE=false
while [[ "$1" == --* ]]; do
    case "$1" in
        --standalone)
            STANDALONE=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            # Unknown global option, break and let command handle it
            break
            ;;
    esac
done

# Parse command
if [ $# -eq 0 ]; then
    usage
    exit 1
fi

COMMAND="$1"
shift

# Helper function to run agent in standalone mode
run_agent_standalone() {
    local model="$1"
    shift
    local tasks=("$@")
    
    # --- NEW: expand comma-separated task lists (e.g. "svm,kmeans")
    local expanded_tasks=()
    for t in "${tasks[@]}"; do
        IFS=',' read -ra PARTS <<< "$t"
        for p in "${PARTS[@]}"; do
            if [ -n "$p" ]; then
                expanded_tasks+=("$p")
            fi
        done
    done
    tasks=("${expanded_tasks[@]}")
    # ---

    # Check if dependencies are available or use container
    if ! python3 -c "import orjson; import numpy" 2>/dev/null; then
        # Dependencies not available, check for Singularity
        if [ -f "$PROJECT_ROOT/config.env" ] || [ -f "$PROJECT_ROOT/slurm/run_config.env" ]; then
            echo "🐍 Running agent via Singularity container..."
            # Use algotune.py which handles Singularity
            exec python3 "$SCRIPT_DIR/algotune.py" agent --standalone --model "$model" "${tasks[@]}"
        else
            echo "❌ Dependencies not installed and no Singularity configured."
            echo "   Please run: pip install -e ."
            exit 1
        fi
    else
        # Dependencies available - run directly
        echo "🐍 Running agent in standalone mode..."
        
        # Create temporary CODE_DIR if not set
        if [ -z "$CODE_DIR" ]; then
            export CODE_DIR=$(mktemp -d)
            echo "📁 Created temporary CODE_DIR: $CODE_DIR"
        fi
        
        # --- NEW: prepare summary file so Python can update it
        local reports_dir="$PROJECT_ROOT/reports"
        mkdir -p "$reports_dir"
        export SUMMARY_FILE="$reports_dir/agent_summary.json"
        if [ ! -f "$SUMMARY_FILE" ]; then
            echo "{}" > "$SUMMARY_FILE"
            echo "📄 Initialized summary file at $SUMMARY_FILE"
        fi
        # ---
        
        # Run tasks
        if [ ${#tasks[@]} -eq 0 ]; then
            echo "❌ No tasks specified. Please specify task names."
            echo "   Example: $0 agent --standalone o4-mini svm kmeans"
            exit 1
        fi
        
        for task in "${tasks[@]}"; do
            echo "🎯 Running task: $task"
            python3 -m AlgoTuner.main --model "$model" --task "$task"
        done
    fi
}

case "$COMMAND" in
    "generate")
        # --- NEW: parse --standalone even when it appears after the command
        while [[ "$1" == --* ]]; do
            case "$1" in
                --standalone)
                    STANDALONE=true
                    shift
                    ;;
                *)
                    break
                    ;;
            esac
        done
        # ---
        if [ "$STANDALONE" = true ] || ([ "$HAS_SLURM" = false ] && [ "$STANDALONE" != true ]); then
            echo "🐍 Running baseline generation in standalone mode..."
            exec python3 "$SCRIPT_DIR/algotune.py" timing --standalone "$@"
        else
            echo "🤖 Submitting baseline generation to SLURM..."
            exec "$SCRIPT_DIR/run_algotune.sh" "$@"
        fi
        ;;
    
    "agent")
        # Parse agent-specific options
        while [[ "$1" == --* ]]; do
            case "$1" in
                --standalone)
                    STANDALONE=true
                    shift
                    ;;
                *)
                    break
                    ;;
            esac
        done
        
        if [ $# -lt 1 ]; then
            echo "Error: agent command requires model name"
            echo "Usage: $0 agent [--standalone] <model> [task]..."
            exit 1
        fi
        
        MODEL="$1"
        shift
        
        if [ "$STANDALONE" = true ] || ([ "$HAS_SLURM" = false ] && [ "$STANDALONE" != true ]); then
            run_agent_standalone "$MODEL" "$@"
        else
            echo "🤖 Submitting AI agent jobs to SLURM..."
            exec "$SCRIPT_DIR/submit_agent.sh" "$MODEL" "$@"
        fi
        ;;
    
    "test")
        if [ "$STANDALONE" = true ] || ([ "$HAS_SLURM" = false ] && [ "$STANDALONE" != true ]); then
            echo "🐍 Running tests in standalone mode..."
            exec python3 "$SCRIPT_DIR/algotune.py" test --standalone "$@"
        else
            echo "🤖 Submitting test jobs to SLURM..."
            exec python3 "$SCRIPT_DIR/algotune.py" test "$@"
        fi
        ;;
    
    "list-tasks")
        exec python3 "$SCRIPT_DIR/algotune.py" list-tasks
        ;;
    
    "list-task-lists")
        exec python3 "$SCRIPT_DIR/algotune.py" list-task-lists
        ;;
    
    "--help"|"-h"|"help")
        usage
        exit 0
        ;;
    
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        usage
        exit 1
        ;;
esac