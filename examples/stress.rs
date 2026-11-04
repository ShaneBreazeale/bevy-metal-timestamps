//! A GPU-heavy scene with the GPU frame time logged next to the frame time.
//!
//! `cargo run --release --example stress -- [cubes_per_side]` (default 60 →
//! 216 000 cubes). Raise it until the frame is GPU-bound: the GPU time then
//! tracks the frame time. With the frame capped (vsync), the GPU time shows the
//! headroom the frame time hides.

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_metal_timestamps::{GpuFrameTime, GpuTimestampsPlugin};

fn main() {
    let side: i32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    App::new()
        .add_plugins((
            DefaultPlugins,
            FrameTimeDiagnosticsPlugin::default(),
            GpuTimestampsPlugin,
        ))
        .insert_resource(Side(side))
        .add_systems(Startup, setup)
        .add_systems(Update, (spin, report))
        .run();
}

#[derive(Resource)]
struct Side(i32);

#[derive(Component)]
struct Spin;

fn setup(
    mut commands: Commands,
    side: Res<Side>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(Cuboid::new(0.6, 0.6, 0.6));
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.5, 0.3),
        ..default()
    });
    let n = side.0;
    let root = commands
        .spawn((Transform::default(), Visibility::default(), Spin))
        .id();
    for x in -n / 2..n / 2 {
        for y in -n / 2..n / 2 {
            for z in -n / 2..n / 2 {
                commands.spawn((
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::from_xyz(x as f32, y as f32, z as f32),
                    ChildOf(root),
                ));
            }
        }
    }
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, 3.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(n as f32 * 1.4, n as f32 * 0.8, n as f32 * 1.4).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn spin(time: Res<Time>, mut q: Query<&mut Transform, With<Spin>>) {
    for mut t in &mut q {
        t.rotate_y(time.delta_secs() * 0.2);
    }
}

fn report(
    time: Res<Time<Real>>,
    gpu: Res<GpuFrameTime>,
    store: Res<DiagnosticsStore>,
    mut next: Local<f32>,
) {
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + 1.0;
    let frame = store
        .get(&FrameTimeDiagnosticsPlugin::FRAME_TIME)
        .and_then(|d| d.smoothed());
    match (gpu.supported, gpu.p50_ms(), gpu.p90_ms()) {
        (Some(false), ..) => info!("GPU timestamps unsupported on this adapter"),
        (_, Some(p50), Some(p90)) => info!(
            "frame {:.2} ms | GPU p50 {p50:.2} ms p90 {p90:.2} ms | {} frames measured",
            frame.unwrap_or(f64::NAN),
            gpu.frames
        ),
        _ => info!("waiting for GPU timestamps…"),
    }
}
