# bevy_metal_timestamps

Real per-frame **GPU time** for [Bevy](https://bevyengine.org) 0.20 — including on
**Metal / Apple silicon**, where Bevy's own render diagnostics report CPU time only.

## Why Bevy shows no GPU time on a Mac

`RenderDiagnosticsPlugin` records GPU time with `write_timestamp` *inside* command
encoders and passes. Apple GPUs sample counters only at **stage boundaries**
(`MTLCounterSamplingPoint::AtStageBoundary`; draw/dispatch/blit-boundary sampling is
unsupported), so wgpu exposes on Metal:

| wgpu feature | Apple silicon |
|---|---|
| `TIMESTAMP_QUERY` (start/end of a pass, via the pass descriptor) | yes |
| `TIMESTAMP_QUERY_INSIDE_ENCODERS` | no |
| `TIMESTAMP_QUERY_INSIDE_PASSES` | no |

Bevy's docs say so too: *"Timestamp queries … supported only on Vulkan and DX12. On other
platforms (Metal …) only CPU time will be recorded."*

`cargo run --example adapter_info` prints what your adapter reports.

## How this crate measures it

It uses only what Metal supports — timestamps written at the **beginning / end of a pass**:

1. **Begin marker** — a compute pass in `RenderGraphSystems::Begin` (the frame's first
   command buffer) writes a timestamp at its start.
2. **End marker** — a compute pass after `RenderGraphSystems::Render`, before `Submit` (the
   frame's last command buffer) writes a timestamp at its end.
3. Command buffers execute in submission order, so `end − begin` is the GPU time of the whole
   frame. The pair is resolved in the **next** frame's first command buffer and read back
   asynchronously over a ring of 8 slots: measuring never stalls the frame; values arrive a
   few frames late.

### Two Metal traps (both measured on an M4)

* **An empty pass samples nothing.** Metal drops an encoder with no work, and its
  timestamps read back as 0. Each marker dispatches one no-op compute thread.
* **An end-of-pass sample is not visible to a resolve in the same command buffer.** Resolving
  right after the end marker returned that slot's *previous* end sample — one full ring old
  (the end lagged the begin by exactly 4 frames with 4 slots; 85 % of samples came out
  negative under load). Resolving in the next command buffer: 733/733 valid.

No Bevy fork, no Objective-C, no unsafe. Works anywhere wgpu reports `TIMESTAMP_QUERY` (Metal, Vulkan,
DX12); without it the plugin logs once and does nothing.

## Use

```toml
[dependencies]
bevy_metal_timestamps = { git = "…" }
```

```rust
use bevy::prelude::*;
use bevy_metal_timestamps::{GpuFrameTime, GpuTimestampsPlugin};

App::new()
    .add_plugins((DefaultPlugins, GpuTimestampsPlugin))
    .add_systems(Update, |gpu: Res<GpuFrameTime>| {
        if let (Some(p50), Some(p90)) = (gpu.p50_ms(), gpu.p90_ms()) {
            info!("GPU frame p50 {p50:.2} ms, p90 {p90:.2} ms");
        }
    })
    .run();
```

Also published as the Bevy diagnostic `gpu_frame_time` (ms), so `LogDiagnosticsPlugin`
and any diagnostics overlay pick it up.

Why it matters: under a frame cap (vsync, frame pacing) the frame time is the cap, and the
CPU-side frame cost hides the GPU's share. The GPU time is the real headroom — what a
frame-rate governor should decide on.

## Example

```sh
cargo run --release --example stress -- 60   # 216k cubes; raise until GPU-bound
```

## Compatibility

| bevy_metal_timestamps | Bevy | wgpu |
|---|---|---|
| 0.1 | 0.20 | 30 |

## License

MIT OR Apache-2.0
