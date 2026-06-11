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
                memory::{MemoryRenderBuffer, MemoryRenderBufferRenderElement},
                solid::SolidColorRenderElement,
                Kind,
            },
            gles::GlesRenderer,
            multigpu::{gbm::GbmGlesBackend, GpuManager, MultiRenderer},
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
    utils::{DeviceFd, Transform},
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
    render_node: DrmNode,
    _token: RegistrationToken,
}

struct State {
    session: LibSeatSession,
    gpus: GpuManager<GbmBackend>,
    #[allow(dead_code)]
    primary_gpu: DrmNode,
    backends: HashMap<DrmNode, BackendData>,
    handle: LoopHandle<'static, State>,
    cursor_x: f64,
    cursor_y: f64,
    output_w: i32,
    output_h: i32,
    needs_redraw: bool,
    text_buffer: Option<MemoryRenderBuffer>,
    text_size: Option<(u32, u32)>,
    cursor_buffer: Option<MemoryRenderBuffer>,
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
        needs_redraw: true,
        text_buffer: None,
        text_size: None,
        cursor_buffer: None,
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
                    state.needs_redraw = true;
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

    if let Some((buf, sz)) = create_hello_world_buffer(48.0) {
        state.text_size = Some(sz);
        state.text_buffer = Some(buf);
        tracing::info!("created Hello World text buffer ({}x{})", sz.0, sz.1);
    } else {
        tracing::warn!("failed to create text buffer");
    }
    state.cursor_buffer = create_crosshair_buffer(6, 4, 14);
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
        if state.needs_redraw {
            state.needs_redraw = false;
            redraw(&mut state);
        }
    }
}

fn create_crosshair_buffer(size: u32, thickness: u32, arm: u32) -> Option<MemoryRenderBuffer> {
    let w = (arm * 2 + size) as u32;
    let h = (arm * 2 + size) as u32;
    let mut buf = vec![0u8; (w * h * 4) as usize];
    let stride = w as usize * 4;
    let cx = w / 2;
    let cy = h / 2;

    for y in 0..h {
        for x in 0..w {
            let dx = x.abs_diff(cx);
            let dy = y.abs_diff(cy);
            let on = (dx < thickness && dy < arm + size / 2)
                || (dy < thickness && dx < arm + size / 2);
            if on {
                let i = y as usize * stride + x as usize * 4;
                buf[i] = 255;
                buf[i + 1] = 255;
                buf[i + 2] = 255;
                buf[i + 3] = 255;
            }
        }
    }

    Some(MemoryRenderBuffer::from_slice(
        &buf,
        Fourcc::Abgr8888,
        (w as i32, h as i32),
        1,
        Transform::Normal,
        None,
    ))
}

fn create_hello_world_buffer(font_size: f32) -> Option<(MemoryRenderBuffer, (u32, u32))> {
    use ab_glyph::{point, Font, FontRef, PxScale, ScaleFont};

    let data = include_bytes!("../DejaVuSans.ttf");
    let font = FontRef::try_from_slice(data).ok()?;
    let px_scale = PxScale::from(font_size);
    let scaled = font.as_scaled(px_scale);

    let text = "Hello World";
    let mut glyphs = Vec::new();
    let mut x = 0f32;
    for c in text.chars() {
        let mut g = font.glyph_id(c).with_scale(px_scale);
        let h = scaled.h_advance(g.id);
        g.position = point(x, scaled.ascent());
        glyphs.push(g);
        x += h;
    }

    let w = x.ceil() as u32;
    let h = scaled.height().ceil() as u32;
    if w == 0 || h == 0 {
        return None;
    }

    let mut buf = vec![0u8; (w * h * 4) as usize];
    let stride = w as usize * 4;

    for g in &glyphs {
        if let Some(outline) = font.outline_glyph(g.clone()) {
            outline.draw(|dx, dy, cov| {
                if dx < w && dy < h {
                    let i = dy as usize * stride + dx as usize * 4;
                    let a = (cov * 255.0) as u8;
                    buf[i] = 255;
                    buf[i + 1] = 255;
                    buf[i + 2] = 255;
                    buf[i + 3] = a;
                }
            });
        }
    }

    Some((
        MemoryRenderBuffer::from_slice(
            &buf,
            Fourcc::Abgr8888,
            (w as i32, h as i32),
            1,
            Transform::Normal,
            None,
        ),
        (w, h),
    ))
}

fn redraw(state: &mut State) {
    tracing::info!("redraw at ({:.0}, {:.0})", state.cursor_x, state.cursor_y);
    for backend in state.backends.values_mut() {
        let rn = backend.render_node;
        for (_crtc, sd) in &mut backend.surfaces {
            let mut renderer = match state.gpus.single_renderer(&rn) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("renderer: {e:?}");
                    continue;
                }
            };

            let sz = sd._output.current_mode().unwrap().size;
            let cx = state.cursor_x as i32;
            let cy = state.cursor_y as i32;

            // Build elements (all MemoryRenderBufferRenderElement for uniform type)
            let mut elements: Vec<MemoryRenderBufferRenderElement<MultiRenderer<GbmBackend, GbmBackend>>> = Vec::new();

            // Hello World text (centered)
            if let (Some(buf), Some((tw, _th))) = (&state.text_buffer, state.text_size) {
                let x = (sz.w - tw as i32) / 2;
                let y = sz.h / 4;
                if let Ok(el) = MemoryRenderBufferRenderElement::from_buffer(
                    &mut renderer,
                    (x as f64, y as f64),
                    buf,
                    Some(1.0),
                    None,
                    None,
                    Kind::Unspecified,
                ) {
                    elements.push(el);
                }
            }

            // cursor crosshair
            if let Some(cbuf) = &state.cursor_buffer {
                let cw = 34; // same as create_crosshair_buffer w
                let ch = 34;
                if let Ok(el) = MemoryRenderBufferRenderElement::from_buffer(
                    &mut renderer,
                    (cx as f64 - cw as f64 / 2.0, cy as f64 - ch as f64 / 2.0),
                    cbuf,
                    Some(1.0),
                    None,
                    None,
                    Kind::Unspecified,
                ) {
                    elements.push(el);
                }
            }

            match sd.drm_output.render_frame(
                &mut renderer,
                &elements,
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

    // try to get a render node, fall back to the drm node itself
    let render_node = (|| -> DrmNode {
        if let Ok(display) = unsafe { EGLDisplay::new(gbm.clone()) } {
            if let Ok(egl_dev) = EGLDevice::device_for_display(&display) {
                let render = egl_dev
                    .try_get_render_node()
                    .ok()
                    .flatten()
                    .unwrap_or(node);
                let _ = state.gpus.as_mut().add_node(render, gbm.clone());
                return render;
            }
        }
        // fallback: register the drm node itself
        let _ = state.gpus.as_mut().add_node(node, gbm.clone());
        node
    })();

    let allocator = GbmAllocator::new(
        gbm.clone(),
        GbmBufferFlags::RENDERING | GbmBufferFlags::SCANOUT,
    );

    let exporter = GbmFramebufferExporter::new(gbm, Some(render_node));

    let color_formats = [Fourcc::Abgr8888, Fourcc::Argb8888];
    let render_formats = {
        let mut renderer = state.gpus.single_renderer(&render_node)?;
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
    render_node: DrmNode,
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

        let mut renderer = match state.gpus.single_renderer(&render_node) {
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
