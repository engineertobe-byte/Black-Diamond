#!/bin/bash
set -e

case "$1" in
    test)
        echo "🔬 Running tests..."
        pytest tests/ -q
        ;;
    example)
        shift
        if [ -z "$1" ]; then
            echo "Usage: example <name>"
            exit 1
        fi
        script="examples/$1.py"
        if [ ! -f "$script" ]; then
            echo "Example '$1' not found."
            exit 1
        fi
        echo "📖 Running example: $1"
        python "$script"
        ;;
    shell)
        echo "🐚 Opening Python shell..."
        python
        ;;
    start)
        shift
        echo "🚀 Starting Black Diamond Launcher..."
        python -m black_diamond.launcher "$@"
        ;;
    version)
        python -c "from black_diamond import __version__; print(__version__)"
        ;;
    *)
        exec "$@"
        ;;
esac
