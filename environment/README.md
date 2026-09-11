# 实验环境

- `core-versions.json` / `requirements.core.txt`：整理时从科研虚拟环境读取的主要库准确版本。
- `requirements.server-freeze.txt`：整理时完整pip freeze；包含平台专属构建和可能的本地安装地址，不是便携lock。
- `requirements.original-freeze.txt`：最早记录的freeze，作为历史证据单独保留，不当成最终依赖列表。

Python3.12、Linux、RTX5090；实际硬件/库信息亦见 `results/overview/provenance.json`。先匹配PyTorch/CUDA，再在隔离环境补齐依赖；本次没有在干净机器完整重装/重训。原env脚本保存在experiments各阶段；不要复制用户登录设置、代理密码或令牌进版本库。
