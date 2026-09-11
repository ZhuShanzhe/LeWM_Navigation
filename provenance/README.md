# 来源与一致性

- `publication_manifest.json` 将仓库路径映射回原工作目录相对路径，并记录大小/SHA256。被选入的历史文件逐字节保留，未重写绝对路径。
- `upstream.json` 记录作者仓库及固定提交；原MIT许可证保留在third_party。
- `excluded-artifacts.json` 说明整类未遍历目录和其他未上传大文件；不是数据下载凭据或完整数据清单。
- `publication_checks.json` 记录本次整理的静态验证、逐例复算和恢复检查结果；不能等同于全新GPU训练复现。

仓库新增README和工具不属于历史源文件哈希。原报告内绝对路径指向服务器归档；通过manifest定位仓库副本。冻结协议里的原源文件哈希不可随意换成整理版路径的哈希。
