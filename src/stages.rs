//! Per-stage GPU time for a 3D view (feature `stages`).
//!
//! Marks go between Bevy's 3D stages; each reads what the stage before it wrote,
//! so it cannot run until that stage has finished.

use std::collections::VecDeque;

use bevy::core_pipeline::core_3d::{main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::schedule::{Core3d, Core3dSystems};
use bevy::pbr::{LATE_SHADOW_PASS, ViewShadowBindings, per_view_shadow_pass, shared_shadow_pass};
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::renderer::{RenderContext, RenderDevice, ViewQuery};
use bevy::render::view::ViewTarget;

use crate::{ExtraMarks, HISTORY, Shared, Timer};

/// Span order in [`GpuStageTimes`]. `shadows` runs from the frame's start, so it
/// includes culling and any compute before the shadow passes.
pub const STAGE_NAMES: [&str; 5] = ["shadows", "opaque", "transparent", "post", "rest"];

/// Main-world result of [`GpuStageTimesPlugin`].
#[derive(Resource, Debug, Clone, Default)]
pub struct GpuStageTimes {
    /// Recent frames, oldest first: ms per [`STAGE_NAMES`] entry.
    pub history: VecDeque<[f32; 5]>,
}

impl GpuStageTimes {
    /// Mean ms per stage over the recent history.
    pub fn mean_ms(&self) -> [f32; 5] {
        let mut mean = [0.0; 5];
        for frame in &self.history {
            for (sum, ms) in mean.iter_mut().zip(frame) {
                *sum += ms / self.history.len() as f32;
            }
        }
        mean
    }
}

/// Adds [`GpuStageTimes`]. Add alongside `GpuTimestampsPlugin`. The marks make each
/// stage wait for the last, so this costs some GPU overlap: for profiling.
pub struct GpuStageTimesPlugin;

impl Plugin for GpuStageTimesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GpuStageTimes>().add_systems(PreUpdate, publish);
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.insert_resource(ExtraMarks(4)).add_systems(
            Core3d,
            (
                mark_shadows
                    .after(per_view_shadow_pass::<LATE_SHADOW_PASS>)
                    .after(shared_shadow_pass::<LATE_SHADOW_PASS>)
                    .before(Core3dSystems::MainPass),
                mark_main::<2>
                    .after(main_opaque_pass_3d)
                    .before(main_transparent_pass_3d),
                mark_main::<3>
                    .after(Core3dSystems::MainPass)
                    .before(Core3dSystems::PostProcess),
                mark_main::<4>.after(Core3dSystems::PostProcess),
            ),
        );
    }
}

fn mark_shadows(
    timer: Option<Res<Timer>>,
    device: Res<RenderDevice>,
    view: ViewQuery<&ViewShadowBindings>,
    mut ctx: RenderContext,
) {
    let Some(timer) = timer else { return };
    let Some(slot) = timer.current else { return };
    let source: &wgpu::TextureView = &view.into_inner().directional_light_depth_texture_view;
    let query = (timer.marks * slot + 1) as u32;
    timer.mark(device.wgpu_device(), ctx.command_encoder(), &timer.depth_marker, &[source], query);
}

fn mark_main<const K: usize>(
    timer: Option<Res<Timer>>,
    device: Res<RenderDevice>,
    view: ViewQuery<&ViewTarget>,
    mut ctx: RenderContext,
) {
    let Some(timer) = timer else { return };
    let Some(slot) = timer.current else { return };
    let source: &wgpu::TextureView = view.into_inner().main_texture_view();
    let query = (timer.marks * slot + K) as u32;
    timer.mark(device.wgpu_device(), ctx.command_encoder(), &timer.marker, &[source], query);
}

fn publish(shared: Res<Shared>, mut times: ResMut<GpuStageTimes>) {
    let frames = std::mem::take(&mut shared.0.lock().unwrap().stages);
    for frame in frames {
        if let Ok(spans) = <[f32; 5]>::try_from(frame) {
            times.history.push_back(spans);
            while times.history.len() > HISTORY {
                times.history.pop_front();
            }
        }
    }
}
