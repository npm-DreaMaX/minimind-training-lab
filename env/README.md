# 分享版环境

按[SHARING_GUIDE.md](../SHARING_GUIDE.md)建立你自己的`.venv`。CPU教学最小依赖见[learning_cpu_requirements.txt](learning_cpu_requirements.txt)。`activate.sh`从自身位置推导仓库，不引用作者账号目录；Windows原生请使用`.venv\Scripts\Activate.ps1`。

本目录其他freeze是原实验依赖证据，不是要求每个学习者安装所有系统包。正式GPU训练需要另行验证CUDA、FLA、精度与恢复；教学入口不会自动训练。
