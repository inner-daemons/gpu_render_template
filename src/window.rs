use std::{
    alloc::GlobalAlloc,
    borrow::BorrowMut,
    collections::{
        hash_map::{DefaultHasher, RandomState},
        BTreeMap, HashMap,
    },
    ffi::OsStr,
    hash::{BuildHasher, BuildHasherDefault},
    io::{stdin, BufRead, BufReader, Read},
    ops::{Deref, DerefMut},
    path::Path,
    sync::{Arc, Mutex},
    thread,
    time::SystemTime,
};

use bytemuck::{Pod, Zeroable};
use std::fs;
use wgpu::{
    naga::{front::glsl::Options, valid::ValidationFlags, FastHashMap},
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BufferBindingType,
    BufferUsages, Device, InstanceFlags, RenderPipeline, ShaderModule, ShaderModuleDescriptor,
    ShaderModuleDescriptorSpirV, ShaderSource, StoreOp, Surface, SurfaceConfiguration,
};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    monitor::VideoMode,
    window::Window,
    window::WindowBuilder,
};

use wgpu::naga::{self, front, valid};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct FragmentUniforms {
    screensize: [f32; 2], // Example: RGBA color
    mousecoords: [f32; 2],
}
unsafe impl Zeroable for FragmentUniforms {}
unsafe impl Pod for FragmentUniforms {}

fn child_stream_to_vec<R>(mut stream: R) -> Arc<Mutex<Vec<String>>>
where
    R: Read + Send + 'static,
{
    let out = Arc::new(Mutex::new(Vec::<String>::new()));
    let vec = out.clone();
    thread::Builder::new()
        .name("child_stream_to_vec".into())
        .spawn(move || {
            let mut buf = [0];
            let mut current = String::new();
            loop {
                match stream.read(&mut buf) {
                    Err(err) => {
                        println!("{}] Error reading from stream: {}", line!(), err);
                        break;
                    }
                    Ok(got) => {
                        if got == 0 {
                            break;
                        } else if got == 1 {
                            if buf[0] as char == '\n' {
                                if current.len() > 0 {
                                    vec.lock().expect("!lock").push(current);
                                    current = String::new();
                                }
                            } else if buf[0] as char == '\r' {
                            } else {
                                current.push(buf[0] as char);
                            }
                        } else {
                            println!("{}] Unexpected number of bytes: {}", line!(), got);
                            break;
                        }
                    }
                }
            }
        })
        .expect("!thread");
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShaderType {
    WGSL,
    GLSL,
    HLSL,
    SPIRV,
    UNKNOWN,
}

struct WindowState<'a> {
    pub surface: wgpu::Surface<'a>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    // The window must be declared after the surface so
    // it gets dropped after it as the surface contains
    // unsafe references to the window's resources.
    pub window: &'a Window,
    pub vertex_shader: wgpu::ShaderModule,
    pub render_pipeline: wgpu::RenderPipeline,
    pub screenshot_pipeline: wgpu::RenderPipeline,
    pub fragment_uniform_buffer: wgpu::Buffer,
    pub fragment_screenshot_uniform_buffer: wgpu::Buffer,
    pub staging_fragment_uniform_buffer: wgpu::Buffer,
    pub bind_group_layout: BindGroupLayout,
    pub fragment_bind_group: wgpu::BindGroup,
    pub fragment_screenshot_bind_group: wgpu::BindGroup,
    pub mouse_coords: [f32; 2],
    pub should_render: bool,
    pub shader_file: Option<String>,
    pub shader_type: ShaderType,
    pub recompile_time: SystemTime,
    pub console_hook: Arc<Mutex<Vec<String>>>,

    pub default_shader: ShaderModule,
}

impl<'a> WindowState<'a> {
    async fn new(window: &'a Window) -> WindowState<'a> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: InstanceFlags::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        //let holder = WindowHolder::from(&window);
        let surface = unsafe { instance.create_surface(window) }.unwrap();
        //let surface = unsafe {instance.create_surface(&window)};

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    // WebGL doesn't support all of wgpu's features, so if
                    // we're building for the web we'll have to disable some.
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();
        let default_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("default.wgsl").into()),
        });

        let width = size.width.min(size.height) as f32;
        let fragment_uniforms: FragmentUniforms = FragmentUniforms {
            screensize: [width, width], // Example: Red color (RGBA),
            mousecoords: [0.0, 0.0],
        };
        let fragment_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("fragment-uniforms"),
            contents: bytemuck::cast_slice(&[fragment_uniforms]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let fragment_screenshot_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("fragment-screenshot-uniforms"),
            contents: bytemuck::cast_slice(&[fragment_uniforms]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        let staging_fragment_uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("fragment-uniforms-staging"),
            contents: bytemuck::cast_slice(&[fragment_uniforms]),
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fragment-uniforms"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<FragmentUniforms>() as wgpu::BufferAddress,
                    ),
                },
                count: None,
            }],
        });

        let fragment_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fragment-uniforms"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(
                    fragment_uniform_buffer.as_entire_buffer_binding(),
                ),
            }],
        });
        let fragment_screenshot_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fragment-uniforms"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(
                    fragment_screenshot_uniform_buffer.as_entire_buffer_binding(),
                ),
            }],
        });

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let vertex_shader_wgsl = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("vertex.wgsl").into()),
        });
        /*println!("Compiling glsl vertex shader");
        let vertex_shader_glsl = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Glsl {
                shader: include_str!("vertex.glsl").into(),
                stage: wgpu::naga::ShaderStage::Vertex,
                defines: HashMap::default()
            }
        });*/
        let (render_pipeline, screenshot_pipeline) = WindowState::make_render_pipeline(
            &device,
            &config,
            &bind_group_layout,
            &vertex_shader_wgsl,
            &default_shader,
        );

        WindowState {
            window,
            surface,
            device,
            queue,
            config,
            size,
            vertex_shader: vertex_shader_wgsl,
            render_pipeline,
            screenshot_pipeline,
            fragment_uniform_buffer,
            fragment_screenshot_uniform_buffer,
            staging_fragment_uniform_buffer,
            bind_group_layout,
            fragment_bind_group,
            fragment_screenshot_bind_group,
            mouse_coords: [0.0, 0.0],
            should_render: true,
            shader_file: None,
            shader_type: ShaderType::UNKNOWN,
            recompile_time: SystemTime::now(),
            console_hook: child_stream_to_vec(stdin()),

            default_shader,
        }
    }
    fn make_render_pipeline(
        device: &Device,
        config: &SurfaceConfiguration,
        bind_group_layout: &BindGroupLayout,
        vertex_shader: &ShaderModule,
        fragment_shader: &ShaderModule,
    ) -> (RenderPipeline, RenderPipeline) {
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render-pipeline-layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
        let mut desc = wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: vertex_shader,
                entry_point: "vs_main", // 1.
                // The buffers that tells it what kind of input to pass (vec4, vec3 etc)
                buffers: &[], // 2.
            },
            fragment: Some(wgpu::FragmentState {
                // 3.
                module: &fragment_shader,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    // 4.
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // PrimitiveTopology -
            // Trianglelist means every 3 points makes a triangle,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // 1.
                strip_index_format: None,
                // Counter clockwise triangles rendering
                front_face: wgpu::FrontFace::Ccw, // 2.
                // Triangles that are not considered facing forward are culled (not included in the render) as specified by CullMode::Back. We'll cover culling a bit more when we cover Buffers.
                cull_mode: //Some(wgpu::Face::Back),
                    None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,                         // 2.
                mask: !0,                         // 3.
                alpha_to_coverage_enabled: false, // 4.
            },
            multiview: None, // 5.
        };
        let render_pipeline = device.create_render_pipeline(&desc);
        desc.fragment.as_mut().unwrap().targets = &[Some(wgpu::ColorTargetState {
            // 4.
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let screenshot_pipeline = device.create_render_pipeline(&desc);
        return (render_pipeline, screenshot_pipeline);
    }
    fn move_mouse(&mut self, new_pos: (f32, f32)) {
        self.mouse_coords = [new_pos.0, new_pos.1];
        let width = self.size.width.min(self.size.height) as f32;
        self.update_fragment_uniform_buffer(
            FragmentUniforms {
                screensize: [width, width],
                mousecoords: [new_pos.0, new_pos.1],
            },
            &self.fragment_uniform_buffer,
        );
        self.should_render = true;
        //println!("Mouse moved to");
    }
    fn update_fragment_uniform_buffer(&self, data: FragmentUniforms, buffer: &wgpu::Buffer) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let buffer_slice = self.staging_fragment_uniform_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Write, |_| {});
            self.device.poll(wgpu::MaintainBase::Wait);

            let data = &[data];
            let content: &[u8] = bytemuck::cast_slice(data);

            let mut buffer_mapped = buffer_slice.get_mapped_range_mut();
            buffer_mapped.copy_from_slice(content);
        }
        encoder.copy_buffer_to_buffer(
            &self.staging_fragment_uniform_buffer,
            0,
            buffer,
            0,
            std::mem::size_of::<FragmentUniforms>() as u64,
        );
        self.staging_fragment_uniform_buffer.unmap();
        self.device.poll(wgpu::Maintain::Wait);

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.size = new_size;
            self.surface.configure(&self.device, &self.config);
            self.update_fragment_uniform_buffer(
                FragmentUniforms {
                    screensize: [new_size.width as f32, new_size.height as f32],
                    mousecoords: self.mouse_coords,
                },
                &self.fragment_uniform_buffer,
            );
            self.should_render = true;
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn get_shadertype(ext: Option<&OsStr>) -> ShaderType {
        if let Some(ext) = ext {
            let s = ext.to_str();
            match s {
                Some(s) => {
                    let s = s.to_ascii_lowercase();
                    return match s.as_str() {
                        "glsl" | "frag" => ShaderType::GLSL,
                        "wgsl" => ShaderType::WGSL,
                        "hlsl" => ShaderType::HLSL,
                        "spirv" => ShaderType::SPIRV,
                        _ => ShaderType::UNKNOWN,
                    };
                }
                None => {
                    return ShaderType::UNKNOWN;
                }
            }
        } else {
            return ShaderType::UNKNOWN;
        }
    }

    fn update(&mut self) {
        let mut should_recompile = false;
        {
            let mut lock = self.console_hook.lock().expect("Failed to lock stdin");
            let value = lock.deref_mut();
            for line in value.iter() {
                let components: Vec<&str> = line.split_ascii_whitespace().collect();
                if components.len() == 0 {
                    continue;
                }
                let command = components[0].to_ascii_lowercase();
                match command.as_str() {
                    "shaderfile" => {
                        if components.len() != 2 {
                            println!("Command shaderfile takes 1 argument");
                            continue;
                        }
                        let path = Path::new(components[1]);
                        if !path.exists() {
                            println!("File does not exist");
                            continue;
                        }
                        self.shader_file = Some(components[1].to_owned());
                        let shadertype = WindowState::get_shadertype(path.extension());
                        self.shader_type = shadertype;
                        if let ShaderType::UNKNOWN = shadertype {
                            println!("Unable to infer shader type. Run the command \"shadertype <glsl | hlsl | wgsl | spirv>\" to manually set the shader type")
                        }
                        should_recompile = true;
                    }
                    "shadertype" => {
                        if components.len() != 2 {
                            println!("Command shadertype takes 1 argument");
                            continue;
                        }
                        let lowercase = components[1].to_ascii_lowercase();
                        match lowercase.as_str() {
                            "glsl" => {
                                self.shader_type = ShaderType::GLSL;
                            }
                            "hlsl" => self.shader_type = ShaderType::HLSL,
                            "wgsl" => self.shader_type = ShaderType::WGSL,
                            "spirv" => self.shader_type = ShaderType::SPIRV,
                            _ => {
                                println!("Unrecognized shader type");
                            }
                        }
                        should_recompile = true;
                    }
                    "screenshot" => {
                        if components.len() != 3 {
                            println!(
                                "Command screenshot takes 2 arguments: dimension and out file"
                            );
                            continue;
                        }
                        let Ok(dim) = components[1].parse::<u32>() else {
                            continue;
                        };
                        let dim = dim.max(1).next_multiple_of(16);
                        let data_size = dim as u64 * dim as u64 * 4;
                        let file = components[2];
                        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                            label: None,
                            size: wgpu::Extent3d {
                                width: dim,
                                height: dim,
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::COPY_SRC,
                            view_formats: &[],
                        });
                        let out_staging_buffer =
                            self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: None,
                                size: data_size,
                                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                                mapped_at_creation: false,
                            });
                        let texture_view =
                            texture.create_view(&wgpu::TextureViewDescriptor::default());

                        self.update_fragment_uniform_buffer(
                            FragmentUniforms {
                                screensize: [dim as f32, dim as f32],
                                mousecoords: self.mouse_coords,
                            },
                            &self.fragment_screenshot_uniform_buffer,
                        );
                        let mut encoder = self.device.create_command_encoder(&Default::default());
                        encoder.copy_buffer_to_buffer(
                            &self.staging_fragment_uniform_buffer,
                            0,
                            &self.fragment_screenshot_uniform_buffer,
                            0,
                            size_of::<FragmentUniforms>() as u64,
                        );
                        {
                            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &texture_view,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                occlusion_query_set: None,
                                timestamp_writes: None,
                            });
                            rp.set_pipeline(&self.screenshot_pipeline);
                            rp.set_bind_group(0, &self.fragment_screenshot_bind_group, &[]);
                            rp.draw(0..6, 0..1);
                        }
                        encoder.copy_texture_to_buffer(
                            wgpu::ImageCopyTexture {
                                texture: &texture,
                                mip_level: 0,
                                origin: wgpu::Origin3d::ZERO,
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::ImageCopyBuffer {
                                buffer: &out_staging_buffer,
                                layout: wgpu::ImageDataLayout {
                                    offset: 0,
                                    // This needs to be a multiple of 256. Normally we would need to pad
                                    // it but we here know it will work out anyways.
                                    bytes_per_row: Some(dim * 4),
                                    rows_per_image: Some(dim),
                                },
                            },
                            wgpu::Extent3d {
                                width: dim,
                                height: dim,
                                depth_or_array_layers: 1,
                            },
                        );
                        self.queue.submit(Some(encoder.finish()));
                        let slice = out_staging_buffer.slice(..);
                        slice.map_async(wgpu::MapMode::Read, |_| ());
                        self.device.poll(wgpu::Maintain::Wait);
                        let range = slice.get_mapped_range();
                        let mut encoder =
                            png::Encoder::new(std::fs::File::create(file).unwrap(), dim, dim);
                        encoder.set_color(png::ColorType::Rgba);
                        let mut png_writer = encoder.write_header().unwrap();
                        png_writer.write_image_data(&range).unwrap();
                        png_writer.finish().unwrap();
                    }
                    _ => {
                        println!("Unrecognized command. Currently supported commands include:\nshaderfile <file>\nshadertype <glsl | hlsl | wgsl | spirv>");
                    }
                }
            }
            value.clear();
        }
        /*if self.window.outer_size() != self.size {
            self.resize(self.window.outer_size());
        }*/
        // Only check if we need to
        if !should_recompile && self.shader_type != ShaderType::UNKNOWN {
            if let Some(path) = self.shader_file.as_mut() {
                let path = Path::new(path);
                if !path.exists() {
                    println!("Shader file no longer exists");
                    self.shader_type = ShaderType::UNKNOWN;
                }
                let data = path.metadata().expect("Unable to get file metadata");
                let mod_time = data
                    .modified()
                    .expect("Unable to get modification time of file");
                if mod_time != self.recompile_time {
                    //println!("Shader save time updated: ");
                    should_recompile = true;
                    self.recompile_time = mod_time;
                }
            }
        }
        if should_recompile {
            match self.update_shader() {
                Some(mod_time) => {
                    // IDK why this is in this match
                    // self.recompile_time = mod_time;
                }
                None => {}
            }
        }
        if self.should_render {
            self.render().expect("Unable to render");
        }
    }
    fn update_shader(&mut self) -> Option<SystemTime> {
        if let ShaderType::UNKNOWN = self.shader_type {
            return None;
        }
        let path = Path::new(self.shader_file.as_mut().unwrap());
        let frag_shader = match self.shader_type {
            ShaderType::WGSL => {
                let out = fs::read_to_string(path).expect("Unable to read text content of file");
                let module = front::wgsl::parse_str(out.as_str());
                match module {
                    Ok(module) => {
                        let valid = valid::Validator::new(
                            valid::ValidationFlags::all(),
                            valid::Capabilities::all(),
                        )
                        .validate(&module);
                        match valid {
                            Ok(_) => {
                                let result: Result<ShaderModule, ()> = Ok({
                                    self.device.create_shader_module(ShaderModuleDescriptor {
                                        label: None,
                                        source: ShaderSource::Wgsl(out.into()),
                                    })
                                });
                                match result {
                                    Ok(value) => Some(value),
                                    Err(_) => None,
                                }
                            }
                            Err(err) => {
                                println!("WGSL validation error: {}", err);
                                None
                            }
                        }
                    }
                    Err(err) => {
                        println!("WGSL parse error: {}", err);
                        None
                    }
                }
            }
            ShaderType::GLSL => {
                let out = fs::read_to_string(path).expect("Unable to read text content of file");
                Some(self.device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: ShaderSource::Glsl {
                        shader: out.into(),
                        stage: wgpu::naga::ShaderStage::Fragment,
                        defines: HashMap::default(),
                    },
                }))
            }
            ShaderType::HLSL => {
                println!("Dependencies do not support hlsl.");
                self.shader_type = ShaderType::UNKNOWN;
                None
            }
            ShaderType::SPIRV => {
                let out = fs::read(path).expect("Unable to read binary contents of file");
                unsafe {
                    Some(
                        self.device
                            .create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
                                label: None,
                                source: wgpu::util::make_spirv_raw(&out[..]),
                            }),
                    )
                }
            }
            _ => None,
        };
        match frag_shader {
            Some(shader) => {
                self.should_render = true;
                (self.render_pipeline, self.screenshot_pipeline) =
                    WindowState::make_render_pipeline(
                        &self.device,
                        &self.config,
                        &self.bind_group_layout,
                        &self.vertex_shader,
                        &shader,
                    );
                return Some(path.metadata().unwrap().modified().unwrap());
            }
            None => {
                self.should_render = true;
                (self.render_pipeline, self.screenshot_pipeline) =
                    WindowState::make_render_pipeline(
                        &self.device,
                        &self.config,
                        &self.bind_group_layout,
                        &self.vertex_shader,
                        &self.default_shader,
                    );
                return None;
            }
        }
    }
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        // submit will accept anything that implements IntoIter
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // This is what @location(0) in the fragment shader targets
                    Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline); // 2.
            render_pass.set_bind_group(0, &self.fragment_bind_group, &[]);
            render_pass.draw(0..6, 0..1); // 3.
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        self.should_render = false;
        Ok(())
    }
}

pub async fn run() {
    //env_logger::init();

    let mut current_mouse_position: (f64, f64) = (0.0, 0.0);

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Gpu Render Template")
        .build(&event_loop)
        .unwrap();
    let mut state = WindowState::new(&window).await;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(|event, target| {
            target.set_control_flow(ControlFlow::Poll);
            match event {
                Event::AboutToWait => {}
                Event::NewEvents(_) => {}
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window.id() => {
                    if !state.input(event) {
                        // UPDATED!
                        match event {
                            WindowEvent::CloseRequested => target.exit(),
                            WindowEvent::KeyboardInput {
                                device_id,
                                event,
                                is_synthetic,
                            } => {}
                            WindowEvent::RedrawRequested => {
                                state.update();
                                match state.render() {
                                    Ok(_) => {}
                                    // Reconfigure the surface if lost
                                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                    // The system is out of memory, we should probably quit
                                    Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                                    Err(e) => eprintln!("{:?}", e),
                                }
                                window.request_redraw();
                            }
                            WindowEvent::Resized(physical_size) => {
                                state.resize(*physical_size);
                            }
                            WindowEvent::ScaleFactorChanged {
                                scale_factor,
                                inner_size_writer,
                            } => {
                                state.resize(state.window.inner_size());
                            }
                            WindowEvent::CursorMoved {
                                device_id: _,
                                position,
                            } => {
                                if position.x != current_mouse_position.0
                                    || position.y != current_mouse_position.1
                                {
                                    state.move_mouse((position.x as f32, position.y as f32));
                                    current_mouse_position = (position.x, position.y);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}
