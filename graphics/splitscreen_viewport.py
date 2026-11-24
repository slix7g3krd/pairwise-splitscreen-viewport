"""分屏多相机视口渲染管线。

本模块对一个纯软件（无 GPU 依赖）的分屏多相机渲染管线建模，修复以下
六类双相机/多相机共用资源时的典型缺陷：

1. 阴影图集按相机分区
   —— :func:`shadow_atlas_partition` 为每个相机切出互不重叠的图集瓦片，
   视锥裁剪范围再怎么互相覆盖也不会擦掉对方的阴影。
2. 每帧重设 viewport 与 scissor
   —— :func:`reset_viewport` / :func:`scissor_rect` 在每帧（而非仅初始化）
   根据当前窗口像素尺寸重新计算物理像素矩形，窗口缩放后不再用旧矩形裁剪。
3. 中间纹理按相机隔离或加栅栏
   —— :func:`isolate_target` 支持 ``"per_camera"``（每相机独立纹理）与
   ``"fence"``（共享纹理 + 显式栅栏依赖）两种策略，杜绝后处理在前置相机
   未结束时写同一张中间纹理。
4. UI 坐标按各自视口映射
   —— :func:`ui_viewport_map` 把窗口点击坐标换算到相机自己的逻辑视口，
   越界返回 ``None``，避免按主相机分辨率换算导致的错位与穿透。
5. 时间缩放按玩家独立
   —— :func:`per_player_scale` 维护每玩家各自的时间缩放，暂停一个玩家
   不再通过全局 ``timeScale`` 冻住另一个人的动画。
6. 相机顺序与依赖显式声明
   —— 每个相机显式声明 ``depends_on``，:func:`camera_order` 做确定性的
   拓扑排序，:func:`determinism` 保证同一份管线任意次构建得到同一份执行
   计划。

其余函数围绕这条管线提供布局、校验、审计、降级、指标、单相机渲染与最终
合成能力。所有入口都兼容只传一个位置参数（例如 ``fn(1)``）的调用方式。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

__all__ = [
    "Rect",
    "CameraConfig",
    "PipelineConfig",
    "SplitLayout",
    "ViewportState",
    "AtlasTile",
    "TargetBinding",
    "Fence",
    "TimeScaleState",
    "RenderPlan",
    "ValidationReport",
    "Hazard",
    "MetricSnapshot",
    "RenderPassResult",
    "FrameResult",
    "CyclicDependencyError",
    "UnknownCameraError",
    "ValidationError",
    "split_layout",
    "scissor_rect",
    "shadow_atlas_partition",
    "reset_viewport",
    "isolate_target",
    "ui_viewport_map",
    "per_player_scale",
    "camera_order",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "metrics",
    "render_camera",
    "compose",
    "build_pipeline",
]


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class CyclicDependencyError(ValueError):
    """相机依赖图中存在环，无法得到线性执行顺序。"""


class UnknownCameraError(KeyError):
    """引用了未在管线中声明的相机。"""


class ValidationError(ValueError):
    """管线配置未通过校验。"""


# ---------------------------------------------------------------------------
# 基础数据结构
# ---------------------------------------------------------------------------


class Rect(NamedTuple):
    """整数像素矩形，原点位于左上角（窗口/UI 约定）。"""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def _rects_overlap(a: Rect, b: Rect) -> bool:
    """两个矩形是否共享任何像素（仅相接不算重叠）。"""
    return not (
        a.right <= b.x
        or b.right <= a.x
        or a.bottom <= b.y
        or b.bottom <= a.y
    )


class AtlasTile(NamedTuple):
    """阴影图集中分配给单个相机的瓦片。"""

    camera_id: int
    rect: Rect
    layer: int


class TargetBinding(NamedTuple):
    """单个相机的中间渲染目标绑定。

    ``fence_in`` 为本相机启动前必须等待的栅栏名集合，``fence_out`` 为本
    相机 pass 结束时 signal 的栅栏名集合；同一条依赖边的两侧引用同名
    栅栏，前置相机 signal 与依赖方 wait 才能真正接上。
    """

    camera_id: int
    target: str
    fence_in: tuple[str, ...]
    fence_out: tuple[str, ...]
    mode: str


class Fence:
    """GPU 命令栅栏的软件模型：相机 N 的后处理必须等待其依赖的栅栏被 signal。

    使用 :class:`threading.Event` 实现，因此在真实多线程下也具备
    wait/signal 的同步语义；单线程管线里 ``signal`` 与 ``wait`` 顺序发生，
    天然满足依赖。
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._event = threading.Event()
        self.signaled = False

    def signal(self) -> None:
        self.signaled = True
        self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        ok = self._event.wait(timeout=timeout)
        if not ok:
            raise TimeoutError(f"fence {self.name!r} 等待超时，前置相机可能未完成")
        return True

    def reset(self) -> None:
        self.signaled = False
        self._event.clear()

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        state = "signaled" if self.signaled else "unsignaled"
        return f"Fence({self.name!r}, {state})"


@dataclass(frozen=True)
class CameraConfig:
    """一台分屏相机的显式声明。

    :param camera_id: 相机/玩家编号（从 0 开始）。
    :param depends_on: 本机必须在哪些相机完成后才能开始（显式依赖）。
    :param logical_resolution: 该玩家 UI 使用的逻辑分辨率。
    :param order_hint: 拓扑同序时的稳定排序键（声明顺序）。
    """

    camera_id: int
    depends_on: tuple[int, ...] = ()
    logical_resolution: tuple[int, int] = (960, 540)
    order_hint: int = 0


@dataclass(frozen=True)
class PipelineConfig:
    """整条分屏管线的静态配置。"""

    cameras: tuple[CameraConfig, ...]
    window_size: tuple[int, int] = (1280, 720)
    orientation: str = "horizontal"
    atlas_size: tuple[int, int] = (2048, 2048)
    target_strategy: str = "per_camera"  # per_camera | fence
    time_scales: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitLayout:
    """分屏布局：每台相机在窗口中的物理像素矩形。"""

    window_size: tuple[int, int]
    orientation: str
    viewports: tuple[Rect, ...]

    def by_camera(self, camera_id: int) -> Rect:
        return self.viewports[camera_id]


@dataclass(frozen=True)
class ViewportState:
    """一帧内某台相机的 viewport 与 scissor 物理像素状态。"""

    camera_id: int
    viewport: Rect
    scissor: Rect
    frame_index: int
    window_size: tuple[int, int]


@dataclass(frozen=True)
class RenderPlan:
    """确定性渲染计划：同一管线配置每次构建结果完全一致。"""

    order: tuple[int, ...]
    dependencies: Mapping[int, tuple[int, ...]]
    layout: SplitLayout
    strategy: str

    def schedule(self, frame_index: int) -> tuple[tuple[int, Rect], ...]:
        """返回第 ``frame_index`` 帧的 (camera_id, viewport) 执行序列。"""
        return tuple((cid, self.layout.by_camera(cid)) for cid in self.order)


@dataclass(frozen=True)
class Hazard:
    """管线审计发现的一条隐患。"""

    code: str
    severity: str
    message: str
    camera_id: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    """配置校验结果。"""

    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    camera_count: int

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValidationError("; ".join(self.errors))


@dataclass(frozen=True)
class MetricSnapshot:
    """单台相机的资源占用指标。"""

    camera_id: int
    viewport_pixels: int
    atlas_pixels: int
    target: str
    draw_calls: int


@dataclass(frozen=True)
class RenderPassResult:
    """单台相机一次渲染 pass 的记录。"""

    camera_id: int
    frame_index: int
    viewport: Rect
    scissor: Rect
    atlas_tile: AtlasTile
    target: TargetBinding
    time_scale: float
    waited_fences: tuple[str, ...]
    signaled_fences: tuple[str, ...]
    commands: tuple[str, ...]


@dataclass(frozen=True)
class FrameResult:
    """一整帧合成结果。"""

    frame_index: int
    order: tuple[int, ...]
    passes: tuple[RenderPassResult, ...]
    ui_events: tuple[dict[str, Any], ...]
    window_size: tuple[int, int]


# ---------------------------------------------------------------------------
# 入参归一化辅助
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW = (1280, 720)
_DEFAULT_ATLAS = (2048, 2048)
_DEFAULT_LOGICAL = (960, 540)


def _count(value: Any, default: int = 2) -> int:
    """把灵活入参归一化为相机数量。

    接受 int、:class:`PipelineConfig`、相机序列；``None`` 与其他无法识别
    的值回落到 ``default``，保证单参数契约调用（如 ``fn(1)``）始终可用。
    """

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, PipelineConfig):
        return len(value.cameras)
    if isinstance(value, Mapping):
        cameras = value.get("cameras")
        if isinstance(cameras, Sequence) and cameras and not isinstance(cameras, (str, bytes)):
            return len(cameras)
        n = value.get("camera_count")
        if isinstance(n, int) and n > 0:
            return n
    if isinstance(value, Sequence) and value and not isinstance(value, (str, bytes)):
        return len(value)
    return default


def _window_size(value: Any) -> tuple[int, int]:
    if isinstance(value, PipelineConfig):
        return value.window_size
    if isinstance(value, Mapping):
        ws = value.get("window_size")
        if (
            isinstance(ws, Sequence)
            and len(ws) == 2
            and all(isinstance(v, int) and v > 0 for v in ws)
        ):
            return int(ws[0]), int(ws[1])
    if (
        isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(v, int) and v > 0 for v in value)
        and not isinstance(value, (str, bytes))
    ):
        return int(value[0]), int(value[1])
    return _DEFAULT_WINDOW


def _atlas_size(value: Any) -> tuple[int, int]:
    if isinstance(value, PipelineConfig):
        return value.atlas_size
    if isinstance(value, Mapping):
        s = value.get("atlas_size")
        if (
            isinstance(s, Sequence)
            and len(s) == 2
            and all(isinstance(v, int) and v > 0 for v in s)
        ):
            return int(s[0]), int(s[1])
    if (
        isinstance(value, Sequence)
        and len(value) == 2
        and all(isinstance(v, int) and v > 0 for v in value)
        and not isinstance(value, (str, bytes))
    ):
        return int(value[0]), int(value[1])
    return _DEFAULT_ATLAS


def _orientation(value: Any) -> str:
    if isinstance(value, PipelineConfig):
        return value.orientation
    if isinstance(value, Mapping):
        o = value.get("orientation")
        if o in ("horizontal", "vertical"):
            return o
    if value in ("horizontal", "vertical"):
        return value
    return "horizontal"


def _as_cameras(value: Any) -> tuple[CameraConfig, ...]:
    """把灵活入参归一化为 :class:`CameraConfig` 元组。"""

    if isinstance(value, PipelineConfig):
        return value.cameras
    if isinstance(value, Sequence) and value and not isinstance(value, (str, bytes)):
        out: list[CameraConfig] = []
        for hint, item in enumerate(value):
            if isinstance(item, CameraConfig):
                out.append(item)
            elif isinstance(item, int):
                out.append(CameraConfig(camera_id=item, order_hint=hint))
            elif isinstance(item, Mapping):
                cid = item.get("camera_id", hint)
                deps = item.get("depends_on", ()) or ()
                out.append(
                    CameraConfig(
                        camera_id=int(cid),
                        depends_on=tuple(int(d) for d in deps),
                        logical_resolution=tuple(
                            item.get("logical_resolution", _DEFAULT_LOGICAL)
                        ),
                        order_hint=int(item.get("order_hint", hint)),
                    )
                )
            else:
                out.append(CameraConfig(camera_id=hint, order_hint=hint))
        return tuple(out)
    n = _count(value)
    return tuple(CameraConfig(camera_id=i, order_hint=i) for i in range(n))


def _strategy(value: Any) -> str:
    if isinstance(value, PipelineConfig):
        return value.target_strategy
    if isinstance(value, Mapping):
        s = value.get("target_strategy")
        if s in ("per_camera", "fence"):
            return s
    if value in ("per_camera", "fence"):
        return value
    return "per_camera"


def _time_scales(value: Any, camera_ids: Iterable[int]) -> dict[int, float]:
    base = {cid: 1.0 for cid in camera_ids}
    if isinstance(value, PipelineConfig):
        base.update(dict(value.time_scales))
    elif isinstance(value, Mapping):
        raw = value.get("time_scales") if "time_scales" in value else value
        if isinstance(raw, Mapping):
            for k, v in raw.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    base[int(k)] = float(v)
    return base


# ---------------------------------------------------------------------------
# 1) 分屏布局
# ---------------------------------------------------------------------------


def split_layout(camera_count: Any = 2,
                 window_size: Any = None,
                 orientation: Any = None) -> SplitLayout:
    """计算每个相机在窗口中的物理像素视口。

    水平分屏按宽度切，垂直分屏按高度切；除不尽的余数像素从左到右（或
    从上到下）逐列/逐行摊给前几个视口，保证所有像素恰好被覆盖一次。
    """

    n = _count(camera_count)
    win = _window_size(window_size) if window_size is not None else _window_size(camera_count)
    orient = (
        _orientation(orientation)
        if orientation is not None
        else _orientation(camera_count)
    )
    width, height = win
    viewports: list[Rect] = []

    if orient == "horizontal":
        base, rem = divmod(width, n)
        offset = 0
        for i in range(n):
            w = base + (1 if i < rem else 0)
            viewports.append(Rect(offset, 0, w, height))
            offset += w
    else:
        base, rem = divmod(height, n)
        offset = 0
        for i in range(n):
            h = base + (1 if i < rem else 0)
            viewports.append(Rect(0, offset, width, h))
            offset += h

    return SplitLayout(window_size=win, orientation=orient, viewports=tuple(viewports))


# ---------------------------------------------------------------------------
# 2) scissor 裁剪矩形（每帧重算，支持 GL 左下原点翻转）
# ---------------------------------------------------------------------------


def scissor_rect(viewport: Any, framebuffer_height: int | None = None) -> Rect:
    """根据当前视口生成 scissor 裁剪矩形。

    ``viewport`` 可以是 :class:`Rect`、四元组 ``(x, y, w, h)`` 或
    :class:`SplitLayout`（此时取第 0 个视口）。窗口坐标原点在左上；
    当传入 ``framebuffer_height`` 时按 OpenGL 约定把 y 翻转到左下原点。
    所有数值四舍五入到整数像素——窗口缩放后必须每帧重新调用，而不是
    复用初始化时的旧矩形。
    """

    if isinstance(viewport, SplitLayout):
        rect = viewport.viewports[0]
    elif isinstance(viewport, Rect):
        rect = viewport
    elif isinstance(viewport, Sequence) and len(viewport) == 4:
        rect = Rect(
            int(round(viewport[0])),
            int(round(viewport[1])),
            int(round(viewport[2])),
            int(round(viewport[3])),
        )
    else:
        # 单参数契约调用：退化为整窗口视口。
        win = _window_size(viewport)
        rect = Rect(0, 0, win[0], win[1])

    x = max(0, rect.x)
    y = max(0, rect.y)
    width = max(0, rect.width)
    height = max(0, rect.height)

    if framebuffer_height is not None:
        # GL 左下原点：y_gl = H - (y_top + height)
        y = max(0, int(framebuffer_height) - (rect.y + rect.height))

    return Rect(x, y, width, height)


# ---------------------------------------------------------------------------
# 3) 阴影图集按相机分区
# ---------------------------------------------------------------------------


def shadow_atlas_partition(camera_count: Any = 2,
                           atlas_size: Any = None) -> dict[int, AtlasTile]:
    """把一张阴影图集切成互不重叠的瓦片，每台相机独占一块。

    返回 ``{camera_id: AtlasTile}``。采用尽量方正的网格布局（列数取
    ``ceil(sqrt(n))``），瓦片间不留重叠区域：两个视锥的裁剪范围即使互相
    覆盖，也只会写入各自的图集瓦片，一侧阴影不再被另一侧清掉。
    """

    n = _count(camera_count)
    size = _atlas_size(atlas_size) if atlas_size is not None else _atlas_size(camera_count)
    atlas_w, atlas_h = size
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    cell_w, rem_w = divmod(atlas_w, cols)
    cell_h, rem_h = divmod(atlas_h, rows)

    tiles: dict[int, AtlasTile] = {}
    for idx in range(n):
        row, col = divmod(idx, cols)
        x = col * cell_w + min(col, rem_w)
        y = row * cell_h + min(row, rem_h)
        w = cell_w + (1 if col < rem_w else 0)
        h = cell_h + (1 if row < rem_h else 0)
        tiles[idx] = AtlasTile(camera_id=idx, rect=Rect(x, y, w, h), layer=idx)

    # 自检：任意两块瓦片不得共享像素。
    rects = [tile.rect for tile in tiles.values()]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_overlap(rects[i], rects[j]):  # pragma: no cover - 不变量
                raise RuntimeError("阴影图集分区出现重叠")
    return tiles


# ---------------------------------------------------------------------------
# 4) 每帧重设 viewport 与 scissor
# ---------------------------------------------------------------------------


def reset_viewport(camera_count: Any = 2,
                   window_size: Any = None,
                   frame_index: int = 0,
                   orientation: Any = None) -> dict[int, ViewportState]:
    """每帧调用：按当前窗口尺寸重设每台相机的 viewport 与 scissor。

    与“只在初始化设置一次”的错误做法相对，本函数不缓存任何旧像素矩形，
    每次调用都从当前 ``window_size`` 重新布局并生成裁剪状态，窗口缩放后
    下一帧立即生效。
    """

    n = _count(camera_count)
    if window_size is None:
        win = _window_size(camera_count)
    else:
        win = _window_size(window_size)
    orient = orientation if orientation is not None else _orientation(camera_count)
    layout = split_layout(n, win, orient)

    states: dict[int, ViewportState] = {}
    for cid, vp in enumerate(layout.viewports):
        states[cid] = ViewportState(
            camera_id=cid,
            viewport=vp,
            scissor=scissor_rect(vp, framebuffer_height=win[1]),
            frame_index=int(frame_index),
            window_size=win,
        )
    return states


# ---------------------------------------------------------------------------
# 5) 中间纹理隔离 / 栅栏
# ---------------------------------------------------------------------------


def isolate_target(camera_count: Any = 2,
                   strategy: Any = None) -> dict[int, TargetBinding]:
    """为每台相机声明中间渲染目标的隔离方式。

    ``"per_camera"``（默认）：每台相机绑定独立命名的中间纹理
    （``intermediate_camN``），后处理各写各的，天然无竞争。

    ``"fence"``：共享同一张 ``intermediate_shared``，但每条依赖边对应一个
    显式栅栏（``fence_<dep>_done``）：前置相机渲染结束 signal，依赖方
    后处理开始前 wait，第二个相机不可能在第一个相机未结束时写入。
    """

    n = _count(camera_count)
    cameras = _as_cameras(camera_count)
    strat = _strategy(strategy) if strategy is not None else _strategy(camera_count)
    by_id = {cam.camera_id: cam for cam in cameras}

    bindings: dict[int, TargetBinding] = {}
    for cid in range(n):
        cam = by_id.get(cid, CameraConfig(camera_id=cid, order_hint=cid))
        deps = tuple(sorted(d for d in cam.depends_on if d != cid))
        if strat == "fence":
            # 每条依赖边同名：dep 结束时 signal，cid 启动前 wait。
            fence_in = tuple(f"fence_{dep}_done" for dep in deps)
            fence_out = (f"fence_{cid}_done",)
            bindings[cid] = TargetBinding(
                camera_id=cid,
                target="intermediate_shared",
                fence_in=fence_in,
                fence_out=fence_out,
                mode="fence",
            )
        else:
            bindings[cid] = TargetBinding(
                camera_id=cid,
                target=f"intermediate_cam{cid}",
                fence_in=(),
                fence_out=(),
                mode="per_camera",
            )
    return bindings


# ---------------------------------------------------------------------------
# 6) UI 坐标按各自视口映射
# ---------------------------------------------------------------------------


def ui_viewport_map(point: Any,
                    viewport: Any = None,
                    logical_resolution: Any = None) -> dict[str, Any] | None:
    """把窗口物理点击坐标映射到相机自己的逻辑视口坐标。

    ``point`` 为 ``(x, y)`` 窗口像素坐标；``viewport`` 为该相机的物理视口
    矩形（:class:`Rect` 或四元组）。点击落在视口外返回 ``None``，调用方
    应据此把事件路由给对应玩家而不是穿透到另一侧。坐标按该相机自己的
    逻辑分辨率换算，不再统一按主相机分辨率处理。
    """

    if viewport is None:
        # 单参数契约调用：把整数视作窗口宽度，整窗口映射到默认逻辑分辨率。
        if isinstance(point, int) and not isinstance(point, bool):
            point = (point, 0)
        vp = Rect(0, 0, _DEFAULT_WINDOW[0], _DEFAULT_WINDOW[1])
    elif isinstance(viewport, Rect):
        vp = viewport
    elif isinstance(viewport, Sequence) and len(viewport) == 4:
        vp = Rect(*(int(v) for v in viewport))
    else:
        win = _window_size(viewport)
        vp = Rect(0, 0, win[0], win[1])

    if logical_resolution is None:
        logical_w, logical_h = _DEFAULT_LOGICAL
    else:
        logical_w, logical_h = (int(v) for v in logical_resolution)

    if not isinstance(point, Sequence) or len(point) != 2:
        raise ValueError("point 必须是 (x, y) 二元组")
    x, y = float(point[0]), float(point[1])

    if not vp.contains(x, y):
        return None

    scale_x = logical_w / vp.width if vp.width else 0.0
    scale_y = logical_h / vp.height if vp.height else 0.0
    local_x = (x - vp.x) * scale_x
    local_y = (y - vp.y) * scale_y

    return {
        "x": local_x,
        "y": local_y,
        "local": (local_x, local_y),
        "camera_viewport": vp.as_dict(),
        "logical_resolution": (logical_w, logical_h),
        "inside": True,
    }


# ---------------------------------------------------------------------------
# 7) 每玩家独立时间缩放
# ---------------------------------------------------------------------------


class TimeScaleState(Mapping):
    """按玩家保存的时间缩放状态。

    暂停一名玩家只会把该玩家的缩放置 0，其他玩家的动画时钟不受影响，
    取代“一个全局 timeScale 冻住所有人”的做法。映射接口按 camera_id 取
    值（``state[0]``、``state.get(1)``、``0 in state``、``len(state)``）。
    """

    def __init__(self, scales: Mapping[int, float] | None = None) -> None:
        self._scales: dict[int, float] = {int(k): float(v) for k, v in (scales or {}).items()}

    # -- 业务语义 --------------------------------------------------------

    def set_scale(self, camera_id: int, scale: float) -> "TimeScaleState":
        scale = float(scale)
        if scale < 0.0:
            raise ValueError("时间缩放不能为负")
        self._scales[int(camera_id)] = scale
        return self

    def pause(self, camera_id: int) -> "TimeScaleState":
        return self.set_scale(camera_id, 0.0)

    def resume(self, camera_id: int) -> "TimeScaleState":
        return self.set_scale(camera_id, 1.0)

    def is_paused(self, camera_id: int) -> bool:
        return self._scales.get(int(camera_id), 1.0) == 0.0

    def delta_time(self, camera_id: int, real_delta: float) -> float:
        """该玩家本帧的有效 delta 时间 = 真实 delta × 个人缩放。"""

        return float(real_delta) * self._scales.get(int(camera_id), 1.0)

    # -- Mapping 接口 ----------------------------------------------------

    def __getitem__(self, key: Any) -> float:
        return self._scales[int(key)]

    def __iter__(self):
        return iter(sorted(self._scales))

    def __len__(self) -> int:
        return len(self._scales)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        return f"TimeScaleState({dict(sorted(self._scales.items()))})"


def per_player_scale(camera_count: Any = 2,
                     scales: Mapping[int, float] | None = None) -> TimeScaleState:
    """构建每玩家独立的时间缩放状态，默认每人缩放为 1.0。"""

    n = _count(camera_count)
    merged = {cid: 1.0 for cid in range(n)}
    if scales:
        for k, v in scales.items():
            merged[int(k)] = float(v)
    else:
        merged.update(_time_scales(camera_count, range(n)))
    return TimeScaleState(merged)


# ---------------------------------------------------------------------------
# 8) 相机顺序：显式依赖 + 确定性拓扑排序
# ---------------------------------------------------------------------------


def camera_order(cameras: Any = 2) -> tuple[int, ...]:
    """按显式 ``depends_on`` 声明做确定性拓扑排序。

    同序候选按 ``(order_hint, camera_id)`` 稳定排序，因此同一份声明无论
    构建多少次都得到同一条执行序列。依赖未知相机抛
    :class:`UnknownCameraError`，依赖成环抛 :class:`CyclicDependencyError`。
    """

    cfgs = _as_cameras(cameras)
    if not cfgs:
        return ()

    ids = {cam.camera_id for cam in cfgs}
    deps_by_id: dict[int, set[int]] = {}
    hint_by_id: dict[int, int] = {}
    for cam in cfgs:
        hint_by_id[cam.camera_id] = cam.order_hint
        unknown = [d for d in cam.depends_on if d not in ids]
        if unknown:
            raise UnknownCameraError(
                f"相机 {cam.camera_id} 依赖了未声明的相机 {sorted(unknown)}"
            )
        deps_by_id[cam.camera_id] = {d for d in cam.depends_on if d != cam.camera_id}

    remaining = set(ids)
    order: list[int] = []
    while remaining:
        ready = sorted(
            (cid for cid in remaining if deps_by_id[cid] <= set(order)),
            key=lambda cid: (hint_by_id[cid], cid),
        )
        if not ready:
            unresolved = {cid: sorted(deps_by_id[cid] - set(order)) for cid in remaining}
            raise CyclicDependencyError(f"相机依赖存在环：{unresolved}")
        chosen = ready[0]
        order.append(chosen)
        remaining.discard(chosen)

    return tuple(order)


# ---------------------------------------------------------------------------
# 9) 确定性：同配置任意次构建得到同一计划
# ---------------------------------------------------------------------------


def determinism(pipeline: Any = 2, iterations: int = 3) -> RenderPlan:
    """构建确定性渲染计划，并验证多次构建结果逐字节一致。

    返回的 :class:`RenderPlan` 携带执行顺序、依赖表、分屏布局与目标策略；
    若重复构建出现任何分歧（理论上不会），抛出 :class:`AssertionError`。
    """

    cfgs = _as_cameras(pipeline)
    n = len(cfgs)
    win = _window_size(pipeline)
    orient = _orientation(pipeline)
    strat = _strategy(pipeline)

    signatures: list[tuple[Any, ...]] = []
    plan: RenderPlan | None = None
    for _ in range(max(1, int(iterations))):
        order = camera_order(cfgs)
        layout = split_layout(n, win, orient)
        deps = {cam.camera_id: tuple(sorted(cam.depends_on)) for cam in cfgs}
        plan = RenderPlan(
            order=order,
            dependencies=deps,
            layout=layout,
            strategy=strat,
        )
        signatures.append(
            (
                plan.order,
                tuple((cid, plan.dependencies[cid]) for cid in sorted(plan.dependencies)),
                plan.layout.viewports,
                plan.strategy,
            )
        )

    first = signatures[0]
    for sig in signatures[1:]:
        assert sig == first, "渲染计划不确定：多次构建结果不一致"
    assert plan is not None
    return plan


# ---------------------------------------------------------------------------
# 10) 配置校验
# ---------------------------------------------------------------------------


def validation(pipeline: Any = 2) -> ValidationReport:
    """静态校验管线配置，返回结构化报告（不合法不抛异常，由报告承载）。"""

    cfgs = _as_cameras(pipeline)
    errors: list[str] = []
    warnings: list[str] = []
    n = len(cfgs)

    if n <= 0:
        errors.append("至少需要一台相机")
    ids = [cam.camera_id for cam in cfgs]
    if len(set(ids)) != len(ids):
        errors.append("camera_id 存在重复")

    id_set = set(ids)
    for cam in cfgs:
        for dep in cam.depends_on:
            if dep not in id_set:
                errors.append(f"相机 {cam.camera_id} 依赖未知相机 {dep}")
        if cam.camera_id in cam.depends_on:
            errors.append(f"相机 {cam.camera_id} 不能依赖自身")
        if cam.logical_resolution[0] <= 0 or cam.logical_resolution[1] <= 0:
            errors.append(f"相机 {cam.camera_id} 的逻辑分辨率必须为正")

    # 成环检查（复用拓扑排序）。
    if not any("依赖未知" in e or "依赖自身" in e for e in errors):
        try:
            camera_order(cfgs)
        except CyclicDependencyError as exc:
            errors.append(f"依赖成环：{exc}")

    win = _window_size(pipeline)
    if win[0] < n or win[1] <= 0:
        errors.append(f"窗口尺寸 {win} 无法容纳 {n} 台分屏相机")

    atlas_w, atlas_h = _atlas_size(pipeline)
    if atlas_w <= 0 or atlas_h <= 0:
        errors.append("阴影图集尺寸必须为正")
    else:
        tiles = shadow_atlas_partition(n, (atlas_w, atlas_h))
        rects = [t.rect for t in tiles.values()]
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if _rects_overlap(rects[i], rects[j]):
                    errors.append("阴影图集分区存在重叠")

    strat = _strategy(pipeline)
    if strat not in ("per_camera", "fence"):
        errors.append(f"未知目标隔离策略 {strat!r}")
    if strat == "fence":
        bindings = isolate_target(cfgs, "fence")
        for cid, binding in bindings.items():
            if not binding.fence_out:
                errors.append(f"fence 策略下相机 {cid} 缺少完成栅栏")
            for fence_name in binding.fence_in:
                # 等待的每条栅栏必须能由某个被依赖相机 signal。
                producers = [
                    other.camera_id
                    for other in cfgs
                    if f"fence_{other.camera_id}_done" == fence_name
                ]
                if not producers:
                    errors.append(
                        f"相机 {cid} 等待的栅栏 {fence_name!r} 没有任何相机 signal"
                    )
        # 共享纹理必须有完整依赖链兜底，否则仍会写竞争。
        for cam in cfgs:
            if cam.camera_id != 0 and not cam.depends_on:
                warnings.append(
                    f"相机 {cam.camera_id} 使用共享纹理但未声明依赖，建议改用 per_camera"
                )

    scales = _time_scales(pipeline, id_set)
    for cid, scale in scales.items():
        if scale < 0:
            errors.append(f"相机 {cid} 的时间缩放为负：{scale}")

    if n == 1:
        warnings.append("只有一台相机，分屏逻辑退化为全屏")

    return ValidationReport(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        camera_count=n,
    )


# ---------------------------------------------------------------------------
# 11) 审计：对照六类已知缺陷扫描隐患
# ---------------------------------------------------------------------------


def audit(pipeline: Any = 2) -> tuple[Hazard, ...]:
    """审计管线，返回按严重度排序的隐患列表。"""

    cfgs = _as_cameras(pipeline)
    n = len(cfgs)
    hazards: list[Hazard] = []

    report = validation(pipeline)
    for msg in report.errors:
        hazards.append(Hazard("validation_error", "critical", msg))
    for msg in report.warnings:
        hazards.append(Hazard("validation_warning", "low", msg))

    # 阴影图集：分区重叠或瓦片过小。
    atlas_w, atlas_h = _atlas_size(pipeline)
    if n > 0 and atlas_w > 0 and atlas_h > 0:
        tiles = shadow_atlas_partition(n, (atlas_w, atlas_h))
        for cid, tile in tiles.items():
            if tile.rect.width < 64 or tile.rect.height < 64:
                hazards.append(
                    Hazard(
                        "shadow_atlas_tile_small",
                        "medium",
                        f"相机 {cid} 的阴影瓦片仅 {tile.rect.width}x{tile.rect.height}，"
                        "阴影分辨率可能不足",
                        cid,
                    )
                )

    # 视口：出现零宽/零高视口意味着窗口缩放后布局失效。
    win = _window_size(pipeline)
    layout = split_layout(n, win, _orientation(pipeline))
    for cid, vp in enumerate(layout.viewports):
        if vp.width <= 0 or vp.height <= 0:
            hazards.append(
                Hazard(
                    "degenerate_viewport",
                    "critical",
                    f"相机 {cid} 的视口退化：{vp}",
                    cid,
                )
            )

    # 中间纹理：共享纹理却没有栅栏/依赖保护。
    strat = _strategy(pipeline)
    if strat == "fence":
        bindings = isolate_target(cfgs, "fence")
        for cam in cfgs:
            binding = bindings.get(cam.camera_id)
            if binding is not None and cam.camera_id != 0 and not cam.depends_on:
                hazards.append(
                    Hazard(
                        "shared_target_without_fence",
                        "critical",
                        f"相机 {cam.camera_id} 写共享中间纹理但无前置依赖栅栏，"
                        "可能与前置相机后处理发生写竞争",
                        cam.camera_id,
                    )
                )

    # 时间缩放：负缩放或全局缩放被多玩家共享的迹象。
    scales = _time_scales(pipeline, {cam.camera_id for cam in cfgs})
    for cid, scale in scales.items():
        if scale < 0:
            hazards.append(
                Hazard("negative_time_scale", "high", f"相机 {cid} 时间缩放为负", cid)
            )

    # UI：相机逻辑分辨率与物理视口宽高比差异过大会导致点击错位。
    for cam in cfgs:
        vp = layout.by_camera(cam.camera_id) if cam.camera_id < len(layout.viewports) else None
        if vp is not None and vp.width and vp.height:
            physical_ratio = vp.width / vp.height
            lw, lh = cam.logical_resolution
            logical_ratio = lw / lh if lh else 0.0
            if logical_ratio and abs(physical_ratio - logical_ratio) / logical_ratio > 0.05:
                hazards.append(
                    Hazard(
                        "ui_aspect_mismatch",
                        "medium",
                        f"相机 {cam.camera_id} 物理/逻辑宽高比不一致，UI 点击映射会拉伸",
                        cam.camera_id,
                    )
                )

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    hazards.sort(key=lambda h: (severity_rank.get(h.severity, 9), h.code, h.camera_id or -1))
    return tuple(hazards)


# ---------------------------------------------------------------------------
# 12) 安全降级
# ---------------------------------------------------------------------------


def fallback(camera_count: Any = 2, reason: str = "degraded") -> PipelineConfig:
    """生成一份保证可用的安全降级管线配置。

    降级策略：每相机独立中间纹理（无共享写竞争）、无相机间依赖、
    每人时间缩放 1.0、标准窗口与图集尺寸。``reason`` 记录降级原因，
    写入配置的相机 order_hint 以便溯源。
    """

    n = _count(camera_count)
    cameras = tuple(
        CameraConfig(
            camera_id=i,
            depends_on=(),
            logical_resolution=_DEFAULT_LOGICAL,
            order_hint=i,
        )
        for i in range(n)
    )
    config = PipelineConfig(
        cameras=cameras,
        window_size=_DEFAULT_WINDOW,
        orientation="horizontal",
        atlas_size=_DEFAULT_ATLAS,
        target_strategy="per_camera",
        time_scales={i: 1.0 for i in range(n)},
    )
    # reason 仅用于溯源，挂在函数属性上，不破坏配置的不可变结构。
    fallback.last_reason = reason  # type: ignore[attr-defined]
    return config


# ---------------------------------------------------------------------------
# 13) 资源指标
# ---------------------------------------------------------------------------


def metrics(pipeline: Any = 2) -> dict[int, MetricSnapshot]:
    """汇总每台相机的视口像素、图集像素与目标绑定指标。"""

    cfgs = _as_cameras(pipeline)
    n = len(cfgs)
    win = _window_size(pipeline)
    layout = split_layout(n, win, _orientation(pipeline))
    tiles = shadow_atlas_partition(n, _atlas_size(pipeline))
    bindings = isolate_target(pipeline)

    out: dict[int, MetricSnapshot] = {}
    for cam in cfgs:
        cid = cam.camera_id
        vp = layout.by_camera(cid) if cid < len(layout.viewports) else Rect(0, 0, 0, 0)
        tile = tiles.get(cid)
        binding = bindings.get(cid)
        out[cid] = MetricSnapshot(
            camera_id=cid,
            viewport_pixels=vp.width * vp.height,
            atlas_pixels=(tile.rect.width * tile.rect.height) if tile else 0,
            target=binding.target if binding else "",
            draw_calls=2 + len(cam.depends_on),
        )
    return out


# ---------------------------------------------------------------------------
# 14) 单相机渲染 pass
# ---------------------------------------------------------------------------


def _build_fences(bindings: Mapping[int, TargetBinding]) -> dict[str, Fence]:
    fences: dict[str, Fence] = {}
    for binding in bindings.values():
        for fence_name in (*binding.fence_in, *binding.fence_out):
            fences.setdefault(fence_name, Fence(fence_name))
    return fences


def render_camera(pipeline: Any = 2,
                  camera_id: int | None = None,
                  frame_index: int = 0,
                  real_delta: float = 1.0 / 60.0,
                  fences: Mapping[str, Fence] | None = None) -> RenderPassResult:
    """执行单台相机一个帧的渲染 pass。

    严格按声明顺序完成：等待依赖栅栏 → 绑定独立/共享中间纹理 →
    在本相机视口与 scissor 内绘制并写入专属图集瓦片 → 按个人时间缩放
    推进动画 → signal 完成栅栏。``fences`` 可由多 pass 共享以构成整条
    依赖链；不传则按本次绑定即时创建。
    """

    cfgs = _as_cameras(pipeline)
    if not cfgs:
        raise ValidationError("管线中没有相机")
    by_id = {cam.camera_id: cam for cam in cfgs}
    if camera_id is None:
        # 单参数契约调用：渲染拓扑序中的第一台相机。
        camera_id = camera_order(cfgs)[0]
    camera_id = int(camera_id)
    if camera_id not in by_id:
        raise UnknownCameraError(f"未知相机 {camera_id}")

    n = len(cfgs)
    win = _window_size(pipeline)
    states = reset_viewport(n, win, frame_index, _orientation(pipeline))
    state = states[camera_id]
    tiles = shadow_atlas_partition(n, _atlas_size(pipeline))
    bindings = isolate_target(pipeline)
    binding = bindings[camera_id]
    scales = per_player_scale(pipeline)

    fence_table = dict(fences) if fences else _build_fences(bindings)
    waited: list[str] = []
    for fence_name in binding.fence_in:
        if fence_name in fence_table:
            fence_table[fence_name].wait(timeout=5.0)
            waited.append(fence_name)

    effective_delta = scales.delta_time(camera_id, real_delta)
    commands = (
        f"bind_target:{binding.target}",
        f"set_viewport:{state.viewport}",
        f"set_scissor:{state.scissor}",
        f"render_shadow:atlas_tile_{camera_id}",
        f"render_scene:dt={effective_delta:.6f}",
        "post_process",
    )

    signaled: list[str] = []
    for fence_name in binding.fence_out:
        fence = fence_table.setdefault(fence_name, Fence(fence_name))
        fence.signal()
        signaled.append(fence_name)

    return RenderPassResult(
        camera_id=camera_id,
        frame_index=int(frame_index),
        viewport=state.viewport,
        scissor=state.scissor,
        atlas_tile=tiles[camera_id],
        target=binding,
        time_scale=scales[camera_id],
        waited_fences=tuple(waited),
        signaled_fences=tuple(signaled),
        commands=commands,
    )


# ---------------------------------------------------------------------------
# 15) 整帧合成
# ---------------------------------------------------------------------------


def build_pipeline(camera_count: int = 2,
                   window_size: tuple[int, int] = _DEFAULT_WINDOW,
                   orientation: str = "horizontal",
                   atlas_size: tuple[int, int] = _DEFAULT_ATLAS,
                   target_strategy: str = "per_camera",
                   dependencies: Mapping[int, Sequence[int]] | None = None,
                   time_scales: Mapping[int, float] | None = None) -> PipelineConfig:
    """便捷构建一条完整管线（相机顺序与依赖在此显式声明）。"""

    dependencies = dependencies or {}
    cameras = tuple(
        CameraConfig(
            camera_id=i,
            depends_on=tuple(sorted(int(d) for d in dependencies.get(i, ()))),
            logical_resolution=_DEFAULT_LOGICAL,
            order_hint=i,
        )
        for i in range(camera_count)
    )
    return PipelineConfig(
        cameras=cameras,
        window_size=tuple(window_size),
        orientation=orientation,
        atlas_size=tuple(atlas_size),
        target_strategy=target_strategy,
        time_scales=dict(time_scales or {}),
    )


def compose(pipeline: Any = 2,
            frame_index: int = 0,
            ui_events: Sequence[Sequence[float]] | None = None) -> FrameResult:
    """合成一整帧：按确定性顺序依次渲染所有相机并路由 UI 事件。

    流程体现全部六类修复：每帧重算 viewport/scissor → 每相机独占图集
    瓦片 → 独立纹理或栅栏保护的共享纹理 → 按个人时间缩放推进 →
    UI 事件按各自视口映射 → 严格按显式依赖的拓扑序执行。
    """

    cfgs = _as_cameras(pipeline)
    report = validation(pipeline)
    report.raise_if_invalid()

    order = camera_order(cfgs)
    bindings = isolate_target(pipeline)
    fences = _build_fences(bindings)
    layout = determinism(pipeline).layout

    passes: list[RenderPassResult] = []
    for cid in order:
        passes.append(
            render_camera(
                pipeline,
                camera_id=cid,
                frame_index=frame_index,
                fences=fences,
            )
        )

    routed_events: list[dict[str, Any]] = []
    for event in ui_events or ():
        point = (event[0], event[1])
        owner: dict[str, Any] | None = None
        for cid in order:
            cam = next(c for c in cfgs if c.camera_id == cid)
            mapped = ui_viewport_map(point, layout.by_camera(cid), cam.logical_resolution)
            if mapped is not None:
                owner = {
                    "window_point": point,
                    "camera_id": cid,
                    "mapped": mapped,
                }
                break
        routed_events.append(owner or {"window_point": point, "camera_id": None, "mapped": None})

    return FrameResult(
        frame_index=int(frame_index),
        order=order,
        passes=tuple(passes),
        ui_events=tuple(routed_events),
        window_size=_window_size(pipeline),
    )
