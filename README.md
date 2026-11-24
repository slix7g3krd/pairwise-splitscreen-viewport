# pairwise-splitscreen-viewport

## 领域

游戏、图形与多媒体

## 主题

分屏多相机视口

## 说明

阴影图集分区、视口重设与相机依赖顺序。

初始状态下 `graphics/splitscreen_viewport.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
