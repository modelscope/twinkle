
**§13.594 续2（defect110 修复加固）**：model-only 分支加 `accelerator.is_fsdp2` 守卫（twinkle **7b997cd6**）——plugin 配置存在不等于 accelerate 处于 FSDP2 模式（CPU-only 进程 distributed_type=NO，prepare() 仅设备放置），守卫保证非 FSDP2 模式行为不变。门重跑 1 passed，ruff 5=5 基线零新增。d110 三臂重跑将对 7b997cd6 戳记。twinkle 本地链：7bc18909（fix＋门）→8ab17d1f（d78-s1 门）→7b997cd6（守卫）；push 候在飞链全 EXIT。
