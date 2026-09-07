"""INGEST 适配层（数据源无关 → 标准 LeRobot v2.1）

把任意采集源的原始产物转换成 innov-dataprep 处理链可消费的标准 v2.1 数据集。
源目录 → adapter 解析 → 标准 v2.1（meta/ + parquet + 视频），之后走既有 01→08。

命名约定（重要，与 pipe/03_clean.py 的 GRIPPER_HINTS 对齐）：
  夹爪/EEF 位姿动作列一律带 gripper 前缀 → 03 的关节限位/跳变/卡死检查自动豁免，
  关节型数据（仿真/遥操作透传）保持 observation.state.{关节名} 原样。

动作空间约定（可配置，默认 UMI/OpenX 式「夹爪位姿 = EEF」）：
  observation.state.gripper_pos_{x,y,z}   3×平移(米)
  observation.state.gripper_quat_{w,x,y,z} 4×单位四元数
  observation.state.gripper_open          1×夹爪开度 0..1
  action.* 同构
"""
