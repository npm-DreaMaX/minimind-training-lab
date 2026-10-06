# Optional Linux/WSL/macOS helper; create .venv in this clone first.
if [ -n "${ZSH_VERSION:-}" ]; then
    _minimind_env_file="${(%):-%x}"
else
    _minimind_env_file="${BASH_SOURCE[0]}"
fi
_minimind_share_root="$(cd -- "$(dirname -- "$_minimind_env_file")/.." && pwd)"
if [ ! -f "$_minimind_share_root/.venv/bin/activate" ]; then
    echo 'Create your environment first; see SHARING_GUIDE.md.' >&2
    return 1
fi
. "$_minimind_share_root/.venv/bin/activate"
export HF_HOME="$_minimind_share_root/data/cache/huggingface"
export TMPDIR="${TMPDIR:-/tmp}/minimind-learning"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"
unset _minimind_env_file _minimind_share_root
