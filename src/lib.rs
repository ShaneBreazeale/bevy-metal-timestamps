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
//! brackets each frame with two tiny marker passes, each dispatching one no-op
//! thread (Metal skips an empty pass, and a skipped pass samples nothing):
//!
//! * a **begin marker** in [`RenderGraphSystems::Begin`] — the first command
//!   buffer of the frame — writing the timestamp at the start of its pass;
//! * an **end marker** after [`RenderGraphSystems::Render`] and before
//!   [`RenderGraphSystems::Submit`] — the last command buffer — writing the
//!   timestamp at the end of its pass.
//!
//! The pair is resolved in the NEXT frame's first command buffer: on Metal an
//! end-of-pass sample is not visible to a resolve in its own command buffer.
//!
//! Command buffers execute in submission order, so `end - begin` is the GPU
//! time of everything the frame submitted in between. Readback is
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
use bevy::render::RenderApp;

/// Readback slots in flight. A slot's end sample is resolved one frame later
/// (see [`collect_and_begin`]) and read back a few frames after that; eight
/// slots keep every frame measured at steady state.
pub const RING: usize = 8;

/// Frames kept in [`GpuFrameTime`] for the percentiles.
pub const HISTORY: usize = 240;

/// Bevy diagnostic carrying each measured frame's GPU time (ms).
pub const GPU_FRAME_TIME: DiagnosticPath = DiagnosticPath::const_new("gpu_frame_time");

/// `QUERY_RESOLVE_BUFFER_ALIGNMENT`: each slot's resolved pair lives at a
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
}

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
    /// One-thread no-op compute pipeline the marker passes dispatch. Metal
    /// skips an EMPTY pass (no timestamps written), and a render pass without
    /// draws only sometimes samples its end (it maps to end-of-fragment).
    marker: wgpu::ComputePipeline,
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
        world.insert_resource(Timer::new(&device, &queue));
        info!("bevy_metal_timestamps: measuring GPU frame time ({RING} readback slots)");
    }
}

impl Timer {
    fn new(device: &RenderDevice, queue: &RenderQueue) -> Self {
        let device = device.wgpu_device();
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("bevy_metal_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: (RING * 2) as u32,
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
                    size: 2 * wgpu::QUERY_SIZE as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
            })
            .collect();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bevy_metal_timestamps_marker"),
            source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
        });
        let marker = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bevy_metal_timestamps_marker"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            query_set,
            resolve,
            slots,
            marker,
            current: None,
            next: 0,
            period_ns: queue.get_timestamp_period(),
        }
    }

    /// Encodes a marker pass: one no-op compute thread (real GPU work, so
    /// Metal keeps the pass) with the requested timestamp writes. Compute
    /// encoders sample at encoder start / end.
    fn marker_pass(&self, encoder: &mut wgpu::CommandEncoder, label: &str, begin: Option<u32>, end: Option<u32>) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.query_set,
                beginning_of_pass_write_index: begin,
                end_of_pass_write_index: end,
            }),
        });
        pass.set_pipeline(&self.marker);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Resolves slot `i`'s (begin, end) pair and copies it to its readback.
    fn resolve_into_readback(&self, encoder: &mut wgpu::CommandEncoder, i: usize) {
        let first = (2 * i) as u32;
        encoder.resolve_query_set(&self.query_set, first..first + 2, &self.resolve, SLOT_STRIDE * i as u64);
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            SLOT_STRIDE * i as u64,
            &self.slots[i].readback,
            0,
            2 * wgpu::QUERY_SIZE as u64,
        );
    }

    /// Converts a resolved (begin, end) tick pair to milliseconds.
    fn elapsed_ms(&self, begin: u64, end: u64) -> Option<f32> {
        frame_ms(begin, end, self.period_ns)
    }
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
    mut pending: ResMut<PendingCommandBuffers>,
) {
    let Some(mut timer) = timer else { return };
    let mut samples = Vec::new();
    for slot in &timer.slots {
        if slot.state.load(Ordering::Acquire) != READY {
            continue;
        }
        let pair = match slot.readback.slice(..).get_mapped_range() {
            Ok(view) => Some((
                u64::from_le_bytes(view[0..8].try_into().unwrap()),
                u64::from_le_bytes(view[8..16].try_into().unwrap()),
            )),
            Err(_) => None,
        };
        slot.readback.unmap();
        slot.state.store(FREE, Ordering::Release);
        if let Some(ms) = pair.and_then(|(begin, end)| timer.elapsed_ms(begin, end)) {
            samples.push(ms);
        }
    }
    if !samples.is_empty() {
        shared.0.lock().unwrap().samples.extend(samples);
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
    let mut encoder = device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bevy_metal_timestamps_begin"),
        });
    // Resolve the slots whose end marker went out last frame. On Metal a
    // counter sampled at the end of a pass is NOT visible to a resolve in the
    // same command buffer — that resolve returned the slot's previous end
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
        timer.marker_pass(&mut encoder, "bevy_metal_timestamps_begin", Some((2 * i) as u32), None);
    }
    pending.push_encoder(encoder, "bevy_metal_timestamps_begin");
}

/// Closes the measurement with the end marker (the last command buffer of
/// the frame) and resolves the pair into the slot's readback buffer.
fn end_and_resolve(
    timer: Option<ResMut<Timer>>,
    device: Res<RenderDevice>,
    mut pending: ResMut<PendingCommandBuffers>,
) {
    let Some(timer) = timer else { return };
    let Some(i) = timer.current else { return };
    let mut encoder = device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bevy_metal_timestamps_end"),
        });
    timer.marker_pass(&mut encoder, "bevy_metal_timestamps_end", None, Some((2 * i + 1) as u32));
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
    }
}
