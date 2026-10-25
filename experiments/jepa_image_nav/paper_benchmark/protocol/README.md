# 任务协议

`common_v1.json` 是四组HSSD测试执行前冻结的原始协议。地点和实例分别计分；其 `official_tracks` 字段列举可选标准任务，不表示本目录完成了类别ObjectNav测试。

`official_gibson_configuration.json` 单独记录Gibson原始验证任务、相机、机器人尺寸、滑动开关、原生计分及检查点。该配置与HSSD受控配置不同，不能用两组分数计算域迁移降幅。原生评测直接注入验证回合，不采用底层模板中的默认训练集路径。

[返回评测入口](../README.md)
