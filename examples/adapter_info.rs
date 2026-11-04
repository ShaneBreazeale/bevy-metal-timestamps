//! Prints which timestamp-query features the render adapter exposes, then exits.
//!
//! On Apple silicon (Metal) expect `TIMESTAMP_QUERY: true` and the two
//! `INSIDE_*` features `false`: counters are sampled only at stage boundaries.

use bevy::prelude::*;
use bevy::render::renderer::{RenderAdapterInfo, RenderDevice};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                visible: false,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Update, report)
        .run();
}

fn report(device: Res<RenderDevice>, info: Res<RenderAdapterInfo>, mut exit: MessageWriter<AppExit>) {
    let f = device.features();
    println!("adapter: {} ({:?})", info.name, info.backend);
    println!(
        "TIMESTAMP_QUERY:                 {}",
        f.contains(wgpu::Features::TIMESTAMP_QUERY)
    );
    println!(
        "TIMESTAMP_QUERY_INSIDE_ENCODERS: {}",
        f.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    );
    println!(
        "TIMESTAMP_QUERY_INSIDE_PASSES:   {}",
        f.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    );
    exit.write(AppExit::Success);
}
