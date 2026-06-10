use smithay::{
    backend::{
        renderer::gles::GlesRenderer,
        winit::{self, WinitEvent},
    },
    output::{Mode, Output, PhysicalProperties, Subpixel},
    reexports::calloop::EventLoop,
    reexports::wayland_server::Display,
    utils::Transform,
};
use std::time::Duration;

fn main() {
    let mut event_loop: EventLoop<()> = EventLoop::try_new().unwrap();
    let display: Display<()> = Display::new().unwrap();
    let mut display_handle = display.handle();

    let (backend, mut winit) =
        winit::init::<GlesRenderer>().expect("Failed to init winit backend");
    let size = backend.window_size();
    let _ = backend;

    let mode = Mode {
        size,
        refresh: 60_000,
    };
    let output = Output::new(
        "winit".to_string(),
        PhysicalProperties {
            size: (0, 0).into(),
            subpixel: Subpixel::Unknown,
            make: "crabwm".into(),
            model: "Winit".into(),
        },
    );
    output.change_current_state(Some(mode), Some(Transform::Normal), None, None);

    loop {
        winit.dispatch_new_events(|event| match event {
            WinitEvent::Resized { size, .. } => {
                let new_mode = Mode {
                    size,
                    refresh: 60_000,
                };
                output.change_current_state(Some(new_mode), None, None, None);
            }
            WinitEvent::CloseRequested => {
                std::process::exit(0);
            }
            _ => {}
        });

        display_handle.flush_clients().unwrap();

        if event_loop.dispatch(Some(Duration::from_millis(1)), &mut ()).is_err() {
            break;
        }
    }
}
