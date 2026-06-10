use std::{
    collections::HashMap,
    path::Path,
    time::Duration,
};

use smithay::{
    backend::{
        allocator::{
            format::FormatSet,
            gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
            Fourcc,
        },
        drm::{
            compositor::FrameFlags,
            exporter::gbm::GbmFramebufferExporter,
            output::{DrmOutput, DrmOutputManager, DrmOutputRenderElements},
            DrmDevice, DrmDeviceFd, DrmEvent, DrmNode,
        },
        egl::{context::ContextPriority, EGLDevice, EGLDisplay},
        input::{InputEvent, KeyboardKeyEvent, PointerButtonEvent, PointerMotionEvent},
        libinput::{LibinputInputBackend, LibinputSessionInterface},
        renderer::{
            element::{
                solid::{SolidColorBuffer, SolidColorRenderElement},
                Kind,
            },
            gles::GlesRenderer,
            multigpu::{gbm::GbmGlesBackend, GpuManager},
            Color32F,
        },
        session::{libseat::LibSeatSession, Event as SessionEvent, Session},
        udev::{primary_gpu, UdevBackend, UdevEvent},
    },
    output::{Mode, Output, PhysicalProperties},
    reexports::{
        calloop::{EventLoop, LoopHandle, RegistrationToken},
        drm::control::{connector, crtc, Device, ModeTypeFlags},
        input::Libinput,
        rustix::fs::OFlags,
        wayland_server::Display,
    },
    utils::DeviceFd,
};

type GbmBackend = GbmGlesBackend<GlesRenderer, DrmDeviceFd>;
type Alloc = GbmAllocator<DrmDeviceFd>;
type Exporter = GbmFramebufferExporter<DrmDeviceFd>;
type OutMgr = DrmOutputManager<Alloc, Exporter, (), DrmDeviceFd>;
type DrmOut = DrmOutput<Alloc, Exporter, (), DrmDeviceFd>;

struct SurfaceData {
    _output: Output,
    drm_output: DrmOut,
}

#[allow(dead_code)]
struct BackendData {
    drm_output_manager: OutMgr,
    surfaces: HashMap<crtc::Handle, SurfaceData>,
    render_node: Option<DrmNode>,
    _token: RegistrationToken,
}

struct State {
    session: LibSeatSession,
    gpus: GpuManager<GbmBackend>,
    primary_gpu: DrmNode,
    backends: HashMap<DrmNode, BackendData>,
    handle: LoopHandle<'static, State>,
    cursor_x: f64,
    cursor_y: f64,
    output_w: i32,
    output_h: i32,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing::Level::WARN.into())
                .from_env_lossy(),
        )
        .init();

    let mut event_loop: EventLoop<State> = EventLoop::try_new().unwrap();
    let _display: Display<State> = Display::new().unwrap();
    let handle = event_loop.handle();

    let (session, session_notifier) = LibSeatSession::new().expect("failed to open session");
    tracing::info!("session: {}", session.seat());

    let gpu_path = primary_gpu(session.seat())
        .ok()
        .flatten()
        .or_else(|| {
            smithay::backend::udev::all_gpus(session.seat())
                .ok()
                .and_then(|mut g| g.pop())
        })
        .expect("no GPU found");
    let primary_gpu = DrmNode::from_path(&gpu_path).expect("invalid GPU node");

    let gpus = GpuManager::new(GbmBackend::with_context_priority(ContextPriority::High))
        .expect("failed to create GPU manager");

    let mut state = State {
        session,
        gpus,
        primary_gpu,
        backends: HashMap::new(),
        handle,
        cursor_x: 100.0,
        cursor_y: 100.0,
        output_w: 1920,
        output_h: 1080,
    };

    state
        .handle
        .insert_source(session_notifier, |event, _, state| match event {
            SessionEvent::PauseSession => {
                tracing::info!("session paused");
                for b in state.backends.values_mut() {
                    b.drm_output_manager.pause();
                }
            }
            SessionEvent::ActivateSession => {
                tracing::info!("session activated");
                for b in state.backends.values_mut() {
                    b.drm_output_manager.activate(false).expect("activate");
                }
            }
        })
        .unwrap();

    let mut libinput =
        Libinput::new_with_udev::<LibinputSessionInterface<LibSeatSession>>(
            state.session.clone().into(),
        );
    libinput.udev_assign_seat("seat0").unwrap();

    state
        .handle
        .insert_source(
            LibinputInputBackend::new(libinput),
            |event, _, state| match event {
                InputEvent::Keyboard { event } => {
                    tracing::info!("key: {:?}", event.key_code());
                }
                InputEvent::PointerButton { event } => {
                    tracing::info!("button: {}", event.button_code());
                }
                InputEvent::PointerMotion { event } => {
                    let p = event.delta();
                    state.cursor_x = (state.cursor_x + p.x).clamp(0.0, state.output_w as f64);
                    state.cursor_y = (state.cursor_y + p.y).clamp(0.0, state.output_h as f64);
                    redraw(state);
                }
                InputEvent::DeviceAdded { device } => {
                    tracing::info!("input: {}", device.name());
                }
                InputEvent::DeviceRemoved { device } => {
                    tracing::info!("removed: {}", device.name());
                }
                _ => {}
            },
        )
        .unwrap();

    let udev = UdevBackend::new("seat0").expect("failed to init udev");

    let first_dev = udev.device_list().next();
    if let Some((dev_id, path)) = first_dev {
        if let Ok(node) = DrmNode::from_dev_id(dev_id) {
            device_added(&mut state, node, path).expect("primary device");
        }
    }
    for (dev_id, path) in udev.device_list() {
        if first_dev
            .map(|(id, _)| id == dev_id)
            .unwrap_or(false)
        {
            continue;
        }
        if let Ok(node) = DrmNode::from_dev_id(dev_id) {
            if let Err(e) = device_added(&mut state, node, path) {
                tracing::warn!("skip {}: {:?}", dev_id, e);
            }
        }
    }

    state
        .handle
        .insert_source(udev, |event, _, state| match event {
            UdevEvent::Added { device_id, path } => {
                if let Ok(node) = DrmNode::from_dev_id(device_id) {
                    if let Err(e) = device_added(state, node, &path) {
                        tracing::error!("add failed: {:?}", e);
                    }
                }
            }
            UdevEvent::Changed { device_id } => {
                if let Ok(node) = DrmNode::from_dev_id(device_id) {
                    device_changed(state, node);
                }
            }
            UdevEvent::Removed { device_id } => {
                if let Ok(node) = DrmNode::from_dev_id(device_id) {
                    device_removed(state, node);
                }
            }
        })
        .unwrap();

    loop {
        event_loop
            .dispatch(Some(Duration::from_millis(16)), &mut state)
            .unwrap();
    }
}

fn redraw(state: &mut State) {
    tracing::info!("redraw at ({:.0}, {:.0})", state.cursor_x, state.cursor_y);
    let rn = state.primary_gpu;
    for backend in state.backends.values_mut() {
        for (_crtc, sd) in &mut backend.surfaces {
            let mut renderer = match state.gpus.single_renderer(&rn) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("renderer: {e:?}");
                    continue;
                }
            };

            let sz = sd._output.current_mode().unwrap().size;
            let bg = SolidColorBuffer::new((sz.w, sz.h), Color32F::new(0.0, 0.2, 0.8, 1.0));
            let bg_el = SolidColorRenderElement::from_buffer(&bg, (0, 0), 1.0, 1.0, Kind::Unspecified);

            let cursor_size = 32;
            let cx = state.cursor_x as i32;
            let cy = state.cursor_y as i32;
            let cur = SolidColorBuffer::new((cursor_size, cursor_size), Color32F::new(1.0, 1.0, 1.0, 1.0));
            let cur_el = SolidColorRenderElement::from_buffer(&cur, (cx, cy), 1.0, 1.0, Kind::Unspecified);

            match sd.drm_output.render_frame(
                &mut renderer,
                &[bg_el, cur_el],
                Color32F::new(0.0, 0.2, 0.8, 1.0),
                FrameFlags::empty(),
            ) {
                Ok(_) => {
                    if let Err(e) = sd.drm_output.commit_frame() {
                        tracing::error!("commit: {e:?}");
                    }
                }
                Err(e) => tracing::error!("render: {e:?}"),
            }
        }
    }
}

fn device_added(
    state: &mut State,
    node: DrmNode,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let fd = state.session.open(
        path,
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOCTTY | OFlags::NONBLOCK,
    )?;
    let fd = DrmDeviceFd::new(DeviceFd::from(fd));

    let (drm, notifier) = DrmDevice::new(fd.clone(), true)?;
    let gbm = GbmDevice::new(fd)?;

    let _token = state
        .handle
        .insert_source(notifier, |event, _, _| match event {
            DrmEvent::VBlank(_crtc) => {}
            DrmEvent::Error(err) => {
                tracing::error!("drm: {err:?}");
            }
        })
        .unwrap();

    let render_node = (|| -> Option<DrmNode> {
        let display = unsafe { EGLDisplay::new(gbm.clone()).ok()? };
        let egl_dev = EGLDevice::device_for_display(&display).ok()?;
        if egl_dev.is_software() {
            return None;
        }
        let render = egl_dev
            .try_get_render_node()
            .ok()
            .flatten()
            .unwrap_or(node);
        state.gpus.as_mut().add_node(render, gbm.clone()).ok()?;
        Some(render)
    })();

    let allocator = match render_node {
        Some(_) => GbmAllocator::new(
            gbm.clone(),
            GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
        ),
        None => {
            let primary = state
                .backends
                .get(&state.primary_gpu)
                .or_else(|| state.backends.values().next())
                .ok_or("no gpu")?;
            primary.drm_output_manager.allocator().clone()
        }
    };

    let exporter = GbmFramebufferExporter::new(gbm, render_node);

    let color_formats = [Fourcc::Abgr8888, Fourcc::Argb8888];
    let render_formats = {
        let rn = render_node.unwrap_or(state.primary_gpu);
        let mut renderer = state.gpus.single_renderer(&rn)?;
        renderer
            .as_mut()
            .egl_context()
            .dmabuf_render_formats()
            .iter()
            .copied()
            .collect::<FormatSet>()
    };

    let mut drm_output_manager = OutMgr::new(
        drm,
        allocator,
        exporter,
        None::<GbmDevice<DrmDeviceFd>>,
        color_formats,
        render_formats,
    );

    let surfaces = init_connectors(state, &mut drm_output_manager, render_node);

    state.backends.insert(
        node,
        BackendData {
            drm_output_manager,
            surfaces,
            render_node,
            _token,
        },
    );
    redraw(state);
    Ok(())
}

fn init_connectors(
    state: &mut State,
    mgr: &mut OutMgr,
    render_node: Option<DrmNode>,
) -> HashMap<crtc::Handle, SurfaceData> {
    let res = match mgr.device().resource_handles() {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("resources: {e:?}");
            return HashMap::new();
        }
    };

    let mut surfaces = HashMap::new();
    let crtc_handles: Vec<crtc::Handle> = res.crtcs().to_vec();
    let mut ci = 0usize;

    for conn in res.connectors() {
        let info = match mgr.device().get_connector(*conn, true) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("connector {conn:?}: {e:?}");
                continue;
            }
        };

        if info.state() != connector::State::Connected || info.modes().is_empty() {
            continue;
        }

        let mode_idx = info
            .modes()
            .iter()
            .position(|m| m.mode_type().contains(ModeTypeFlags::PREFERRED))
            .unwrap_or(0);
        let drm_mode = info.modes()[mode_idx];
        let wl_mode = Mode::from(drm_mode);

        let crtc = match crtc_handles.get(ci) {
            Some(c) => {
                ci += 1;
                *c
            }
            None => {
                tracing::warn!("no CRTC left for {}", info.interface().as_str());
                continue;
            }
        };

        let name = format!("{}-{}", info.interface().as_str(), info.interface_id());
        tracing::info!("connected: {name}");

        let (pw, ph) = info.size().unwrap_or((0, 0));
        let output = Output::new(
            name,
            PhysicalProperties {
                size: (pw as i32, ph as i32).into(),
                subpixel: info.subpixel().into(),
                make: "unknown".into(),
                model: "unknown".into(),
            },
        );
        output.set_preferred(wl_mode);
        output.change_current_state(Some(wl_mode), None, None, None);

        let planes = match mgr.device().planes(&crtc) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("planes: {e:?}");
                continue;
            }
        };

        let rn = render_node.unwrap_or(state.primary_gpu);
        let mut renderer = match state.gpus.single_renderer(&rn) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("renderer: {e:?}");
                continue;
            }
        };

        let drm_output = match mgr.initialize_output::<_, SolidColorRenderElement>(
            crtc,
            drm_mode,
            &[*conn],
            &output,
            Some(planes),
            &mut renderer,
            &DrmOutputRenderElements::default(),
        ) {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!("output init: {e:?}");
                continue;
            }
        };

        // store output size for cursor bounds
        let sz = output.current_mode().unwrap().size;
        state.output_w = sz.w;
        state.output_h = sz.h;

        surfaces.insert(
            crtc,
            SurfaceData {
                _output: output,
                drm_output,
            },
        );
    }

    surfaces
}

fn device_changed(state: &mut State, node: DrmNode) {
    let _ = (state, node);
}

fn device_removed(state: &mut State, node: DrmNode) {
    if let Some(mut backend) = state.backends.remove(&node) {
        backend.surfaces.clear();
    }
}
