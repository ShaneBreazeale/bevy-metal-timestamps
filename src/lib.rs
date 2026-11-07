//! Real per-frame GPU time for Bevy 0.20 — including on Metal.
//!
//! Bevy's `RenderDiagnosticsPlugin` writes its GPU timestamps *inside* command
//! encoders and passes (`write_timestamp`). Apple GPUs only sample counters at
//! **stage boundaries**, so wgpu exposes `TIMESTAMP_QUERY` on Metal but not
//! `TIMESTAMP_QUERY_INSIDE_ENCODERS` / `_INSIDE_PASSES`, and Bevy records CPU
//! time only there.
//!
//! What Metal *does* support is a timestamp at the beginning or end of a pass,
//! requested in the pass descriptor (`ComputePassTimestampWrites`). This crate
//! brackets each frame with two tiny marker passes, each dispatching one thread
//! (Metal skips an empty pass, and a skipped pass samples nothing):
//!
//! * a **begin marker** in [`RenderGraphSystems::Begin`], the first command
//!   buffer of the frame;
//! * an **end marker** after [`RenderGraphSystems::Render`] and before
//!   [`RenderGraphSystems::Submit`], the last command buffer.
//!
//! A marker must DEPEND on the work it measures. Command buffers start in
//! submission order but passes with no shared resource overlap freely, so a
//! free-standing marker ran early: on a saturated GPU the end marker landed right
//! after the begin marker and the frame read ~0 ms. Each marker therefore reads
//! the view's main texture (hazard tracking waits for its writers), writes one
//! shared buffer (which chains the markers in order), and samples at the END of
//! its pass.
//!
//! The pair is resolved in the NEXT frame's first command buffer: on Metal an
//! end-of-pass sample is not visible to a resolve in its own command buffer.
//!
//! `end - begin` is then the GPU time of the frame. Readback is
//! asynchronous over a ring of [`RING`] slots: measuring never stalls a frame,
//! the value just arrives a few frames late. Frames are skipped while every
//! slot is still in flight.
//!
//! Works anywhere wgpu reports `TIMESTAMP_QUERY` (Metal on Apple silicon,
//! Vulkan, DX12). Without it the plugin logs once and does nothing.
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_metal_timestamps::{GpuFrameTime, GpuTimestampsPlugin};
//!
//! fn main() {
//!     App::new()
//!         .add_plugins((DefaultPlugins, GpuTimestampsPlugin))
//!         .add_systems(Update, |gpu: Res<GpuFrameTime>| {
//!             if let Some(ms) = gpu.p50_ms() {
//!                 info!("GPU frame p50 {ms:.2} ms");
//!             }
//!         })
//!         .run();
//! }
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use bevy::diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy::prelude::*;
use bevy::render::renderer::{
    PendingCommandBuffers, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue,
};
use bevy::render::view::ViewTarget;
use bevy::render::RenderApp;

#[cfg(feature = "stages")]
mod stages;
#[cfg(feature = "stages")]
pub use stages::{GpuStageTimes, GpuStageTimesPlugin, STAGE_NAMES};

/// Readback slots in flight. A slot's end sample is resolved one frame later
/// (see [`collect_and_begin`]) and read back a few frames after that; eight
/// slots keep every frame measured at steady state.
pub const RING: usize = 8;

/// Frames kept in [`GpuFrameTime`] for the percentiles.
pub const HISTORY: usize = 240;

/// Bevy diagnostic carrying each measured frame's GPU time (ms).
pub const GPU_FRAME_TIME: DiagnosticPath = DiagnosticPath::const_new("gpu_frame_time");

/// `QUERY_RESOLVE_BUFFER_ALIGNMENT`: each slot's resolved marks live at a
/// multiple of this offset.
const SLOT_STRIDE: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

/// Main-world result.
#[derive(Resource, Debug, Clone, Default)]
pub struct GpuFrameTime {
    /// `Some(false)` when the adapter has no `TIMESTAMP_QUERY`; `None` until
    /// the render world has checked.
    pub supported: Option<bool>,
    /// Most recent measured frame (ms).
    pub last_ms: Option<f32>,
    /// Recent frames, oldest first (ms), at most [`HISTORY`].
    pub history: VecDeque<f32>,
    /// Measured frames since start.
    pub frames: u64,
}

impl GpuFrameTime {
    /// Percentile `q` in 0..=100 of the recent history.
    pub fn percentile_ms(&self, q: f32) -> Option<f32> {
        if self.history.is_empty() {
            return None;
        }
        let mut sorted: Vec<f32> = self.history.iter().copied().collect();
        sorted.sort_by(f32::total_cmp);
        let i = ((sorted.len() - 1) as f32 * (q / 100.0).clamp(0.0, 1.0)).round() as usize;
        Some(sorted[i])
    }

    /// Percentile `q` (0..=100) over only the most recent `last` frames —
    /// for decisions that must follow a scene change quickly (frame pacing).
    pub fn recent_percentile_ms(&self, q: f32, last: usize) -> Option<f32> {
        let skip = self.history.len().saturating_sub(last.max(1));
        let mut sorted: Vec<f32> = self.history.iter().skip(skip).copied().collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(f32::total_cmp);
        let i = ((sorted.len() - 1) as f32 * (q / 100.0).clamp(0.0, 1.0)).round() as usize;
        Some(sorted[i])
    }

    pub fn p50_ms(&self) -> Option<f32> {
        self.percentile_ms(50.0)
    }

    pub fn p90_ms(&self) -> Option<f32> {
        self.percentile_ms(90.0)
    }

    fn push(&mut self, ms: f32) {
        self.last_ms = Some(ms);
        self.frames += 1;
        self.history.push_back(ms);
        while self.history.len() > HISTORY {
            self.history.pop_front();
        }
    }
}

/// Render → main world hand-off.
#[derive(Resource, Clone, Default)]
struct Shared(Arc<Mutex<SharedState>>);

#[derive(Default)]
struct SharedState {
    supported: Option<bool>,
    samples: Vec<f32>,
    #[cfg_attr(not(feature = "stages"), allow(dead_code))]
    stages: Vec<Vec<f32>>,
}

/// Marks per frame beyond begin and end; set by `GpuStageTimesPlugin`.
#[derive(Resource, Clone, Copy)]
struct ExtraMarks(#[cfg_attr(not(feature = "stages"), allow(dead_code))] usize);

const FREE: u8 = 0;
/// Both markers encoded; the end sample is not resolvable yet.
const RECORDED: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;
/// Resolve + copy encoded (in a later command buffer than the end marker).
const RESOLVED: u8 = 4;

struct Slot {
    readback: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

/// Render-world state.
#[derive(Resource)]
struct Timer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<Slot>,
    /// Marks per frame: begin, any stage marks, end.
    marks: usize,
    /// Marker pipelines: one thread reads a texel of the texture it waits on and
    /// stores it in `sink`. Metal skips an EMPTY pass (no timestamps written).
    marker: wgpu::ComputePipeline,
    #[cfg_attr(not(feature = "stages"), allow(dead_code))]
    depth_marker: wgpu::ComputePipeline,
    sink: wgpu::Buffer,
    /// Bound when a frame has no view, so a marker still chains through `sink`.
    fallback: wgpu::TextureView,
    /// Slot measuring the frame being recorded.
    current: Option<usize>,
    next: usize,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
}

/// Adds [`GpuFrameTime`] and the [`GPU_FRAME_TIME`] diagnostic.
pub struct GpuTimestampsPlugin;

impl Plugin for GpuTimestampsPlugin {
    fn build(&self, app: &mut App) {
        let shared = Shared::default();
        app.insert_resource(shared.clone())
            .init_resource::<GpuFrameTime>()
            .register_diagnostic(Diagnostic::new(GPU_FRAME_TIME).with_suffix("ms"))
            .add_systems(PreUpdate, publish);
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.insert_resource(shared).add_systems(
                RenderGraph,
                (
                    collect_and_begin.in_set(RenderGraphSystems::Begin),
                    end_and_resolve
                        .after(RenderGraphSystems::Render)
                        .before(RenderGraphSystems::Submit),
                    request_readback.in_set(RenderGraphSystems::Finish),
                ),
            );
        }
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        let world = render_app.world_mut();
        let (Some(device), Some(queue)) = (
            world.get_resource::<RenderDevice>().cloned(),
            world.get_resource::<RenderQueue>().cloned(),
        ) else {
            return;
        };
        let supported = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        if let Some(shared) = world.get_resource::<Shared>() {
            shared.0.lock().unwrap().supported = Some(supported);
        }
        if !supported {
            warn!("bevy_metal_timestamps: adapter has no TIMESTAMP_QUERY; GPU frame time unavailable");
            return;
        }
        let marks = 2 + world.get_resource::<ExtraMarks>().map_or(0, |m| m.0);
        world.insert_resource(Timer::new(&device, &queue, marks));
        info!("bevy_metal_timestamps: measuring GPU frame time ({RING} readback slots)");
    }
}

impl Timer {
    fn new(device: &RenderDevice, queue: &RenderQueue, marks: usize) -> Self {
        let device = device.wgpu_device();
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bevy_metal_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: (RING * marks) as u32,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bevy_metal_timestamps_resolve"),
            size: SLOT_STRIDE * RING as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..RING)
            .map(|_| Slot {
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bevy_metal_timestamps_readback"),
                    size: marks as u64 * wgpu::QUERY_SIZE as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
            })
            .collect();
        let sink = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bevy_metal_timestamps_sink"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fallback = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("bevy_metal_timestamps_fallback"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let marker = marker_pipeline(
            device,
            "@group(0) @binding(0) var source: texture_2d<f32>;
             @group(0) @binding(1) var<storage, read_write> sink: array<f32>;
             @compute @workgroup_size(1) fn main() {
                 sink[0] = textureLoad(source, vec2<u32>(0u), 0).x;
             }",
        );
        let depth_marker = marker_pipeline(
            device,
            "@group(0) @binding(0) var source: texture_depth_2d_array;
             @group(0) @binding(1) var<storage, read_write> sink: array<f32>;
             @compute @workgroup_size(1) fn main() {
                 sink[0] = textureLoad(source, vec2<u32>(0u), 0, 0);
             }",
        );
        Self {
            query_set,
            resolve,
            slots,
            marks,
            marker,
            depth_marker,
            sink,
            fallback,
            current: None,
            next: 0,
            period_ns: queue.get_timestamp_period(),
        }
    }

    /// Encodes one marker pass, sampling at its end: after the dispatches, so after
    /// the writers of each source. Compute encoders sample at encoder start / end.
    fn mark(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        sources: &[&wgpu::TextureView],
        query: u32,
    ) {
        let groups: Vec<wgpu::BindGroup> = sources
            .iter()
            .map(|source| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bevy_metal_timestamps_marker"),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(source),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.sink.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bevy_metal_timestamps_marker"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(query),
            }),
        });
        pass.set_pipeline(pipeline);
        for group in &groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }

    /// Resolves slot `i`'s marks and copies them to its readback.
    fn resolve_into_readback(&self, encoder: &mut wgpu::CommandEncoder, i: usize) {
        let first = (self.marks * i) as u32;
        encoder.resolve_query_set(
            &self.query_set,
            first..first + self.marks as u32,
            &self.resolve,
            SLOT_STRIDE * i as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            SLOT_STRIDE * i as u64,
            &self.slots[i].readback,
            0,
            self.marks as u64 * wgpu::QUERY_SIZE as u64,
        );
    }

    /// Milliseconds between consecutive marks; `None` for a frame with a mark
    /// unwritten or out of order.
    fn spans_ms(&self, ticks: &[u64]) -> Option<Vec<f32>> {
        ticks
            .windows(2)
            .map(|pair| frame_ms(pair[0], pair[1], self.period_ns))
            .collect()
    }
}

fn marker_pipeline(device: &wgpu::Device, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bevy_metal_timestamps_marker"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bevy_metal_timestamps_marker"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Each view's main texture, or the fallback when nothing renders this frame.
fn frame_sources<'a>(timer: &'a Timer, views: &'a Query<&ViewTarget>) -> Vec<&'a wgpu::TextureView> {
    let sources: Vec<&wgpu::TextureView> = views.iter().map(|v| &**v.main_texture_view()).collect();
    if sources.is_empty() { vec![&timer.fallback] } else { sources }
}

/// `(end - begin)` ticks at `period_ns` per tick, in ms. `None` for an
/// unwritten (zero) or reversed pair.
pub fn frame_ms(begin: u64, end: u64, period_ns: f32) -> Option<f32> {
    if begin == 0 || end <= begin {
        return None;
    }
    Some(((end - begin) as f64 * period_ns as f64 / 1.0e6) as f32)
}

/// Reads finished slots, then opens this frame's measurement with the begin
/// marker (the first command buffer of the frame).
fn collect_and_begin(
    timer: Option<ResMut<Timer>>,
    device: Res<RenderDevice>,
    shared: Res<Shared>,
    views: Query<&ViewTarget>,
    mut pending: ResMut<PendingCommandBuffers>,
) {
    let Some(mut timer) = timer else { return };
    let mut samples: Vec<f32> = Vec::new();
    let mut stages = Vec::new();
    for slot in &timer.slots {
        if slot.state.load(Ordering::Acquire) != READY {
            continue;
        }
        let ticks: Option<Vec<u64>> = match slot.readback.slice(..).get_mapped_range() {
            Ok(view) => Some(
                view.chunks_exact(8)
                    .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
                    .collect(),
            ),
            Err(_) => None,
        };
        slot.readback.unmap();
        slot.state.store(FREE, Ordering::Release);
        if let Some(spans) = ticks.and_then(|ticks| timer.spans_ms(&ticks)) {
            samples.push(spans.iter().sum());
            if spans.len() > 1 {
                stages.push(spans);
            }
        }
    }
    if !samples.is_empty() {
        let mut state = shared.0.lock().unwrap();
        state.samples.extend(samples);
        state.stages.extend(stages);
    }

    // Next free slot; skip this frame if every slot is still in flight.
    timer.current = None;
    for k in 0..RING {
        let i = (timer.next + k) % RING;
        if timer.slots[i].state.load(Ordering::Acquire) == FREE {
            timer.current = Some(i);
            timer.next = (i + 1) % RING;
            break;
        }
    }
    let wgpu_device = device.wgpu_device();
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bevy_metal_timestamps_begin"),
    });
    // Resolve the slots whose end marker went out last frame. On Metal a
    // counter sampled at the end of a pass is NOT visible to a resolve in the
    // same command buffer: that resolve returned the slot's previous end
    // sample, one whole ring old (measured: end lagged begin by ~4 frames).
    // This encoder is the first command buffer of the next frame.
    for (j, slot) in timer.slots.iter().enumerate() {
        if slot.state.load(Ordering::Acquire) != RECORDED {
            continue;
        }
        timer.resolve_into_readback(&mut encoder, j);
        slot.state.store(RESOLVED, Ordering::Release);
    }
    if let Some(i) = timer.current {
        let sources = frame_sources(&timer, &views);
        timer.mark(wgpu_device, &mut encoder, &timer.marker, &sources, (timer.marks * i) as u32);
    }
    pending.push_encoder(encoder, "bevy_metal_timestamps_begin");
}

/// Closes the measurement with the end marker, the last command buffer of the
/// frame.
fn end_and_resolve(
    timer: Option<ResMut<Timer>>,
    device: Res<RenderDevice>,
    views: Query<&ViewTarget>,
    mut pending: ResMut<PendingCommandBuffers>,
) {
    let Some(timer) = timer else { return };
    let Some(i) = timer.current else { return };
    let wgpu_device = device.wgpu_device();
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bevy_metal_timestamps_end"),
    });
    let sources = frame_sources(&timer, &views);
    let query = (timer.marks * i + timer.marks - 1) as u32;
    timer.mark(wgpu_device, &mut encoder, &timer.marker, &sources, query);
    pending.push_encoder(encoder, "bevy_metal_timestamps_end");
    timer.slots[i].state.store(RECORDED, Ordering::Release);
}

/// After submit: map every resolved slot (completes once the GPU is done).
fn request_readback(timer: Option<Res<Timer>>) {
    let Some(timer) = timer else { return };
    for slot in &timer.slots {
        if slot.state.load(Ordering::Acquire) != RESOLVED {
            continue;
        }
        slot.state.store(MAPPING, Ordering::Release);
        let state = slot.state.clone();
        slot.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                state.store(if result.is_ok() { READY } else { FREE }, Ordering::Release);
            });
    }
}

/// Main world: move render-world samples into [`GpuFrameTime`] and the
/// diagnostic.
fn publish(shared: Res<Shared>, mut gpu: ResMut<GpuFrameTime>, mut diagnostics: Diagnostics) {
    let (supported, samples) = {
        let mut state = shared.0.lock().unwrap();
        (state.supported, std::mem::take(&mut state.samples))
    };
    if gpu.supported != supported {
        gpu.supported = supported;
    }
    for ms in samples {
        gpu.push(ms);
        diagnostics.add_measurement(&GPU_FRAME_TIME, || ms as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_pairs_convert_and_reject_unwritten() {
        // 1.5 M ticks at 1 ns = 1.5 ms (Metal reports nanoseconds, period 1).
        assert_eq!(frame_ms(10, 1_500_010, 1.0), Some(1.5));
        // Period scales ticks (e.g. 41.67 ns/tick on some GPUs).
        assert!((frame_ms(1, 1 + 24_000, 41.666_668).unwrap() - 1.0).abs() < 1e-3);
        assert_eq!(frame_ms(0, 100, 1.0), None, "begin never written");
        assert_eq!(frame_ms(500, 400, 1.0), None, "reversed pair");
    }

    #[test]
    fn percentiles_over_history() {
        let mut gpu = GpuFrameTime::default();
        for ms in [4.0, 1.0, 3.0, 2.0, 5.0] {
            gpu.push(ms);
        }
        assert_eq!(gpu.p50_ms(), Some(3.0));
        assert_eq!(gpu.percentile_ms(100.0), Some(5.0));
        assert_eq!(gpu.last_ms, Some(5.0));
        for _ in 0..HISTORY {
            gpu.push(1.0);
        }
        assert_eq!(gpu.history.len(), HISTORY);
        // Recent window ignores the old samples.
        gpu.push(9.0);
        assert_eq!(gpu.recent_percentile_ms(100.0, 1), Some(9.0));
        assert_eq!(gpu.recent_percentile_ms(50.0, 3), Some(1.0));
    }
}
