use eframe::egui;
use egui::{ColorImage, TextureHandle};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::thread;
use std::time::Instant;
use crossbeam_channel::{bounded, Receiver, Sender};

const G: f64 = 9.81; // Gravitational acceleration (m/s²)

#[derive(Clone, Copy, PartialEq)]
struct PendulumParams {
    m1: f64, // Mass of first pendulum (kg)
    m2: f64, // Mass of second pendulum (kg)
    l1: f64, // Length of first pendulum (m)
    l2: f64, // Length of second pendulum (m)
}

#[derive(Clone, Copy, PartialEq)]
struct SimulationConfig {
    dt: f64,      // Time step for numerical integration (s)
    steps: usize, // Number of RK4 integration steps per pixel
}

#[derive(Clone, Copy, PartialEq)]
struct RenderConfig {
    width: u32,
    height: u32,
    preview_quality: bool, // If true, use lower quality for real-time preview
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum ColorScheme {
    RedPurple,
    BlueGreen,
    Rainbow,
    Fire,
    Ocean,
    Sunset,
    Custom,
}

#[derive(Clone, Copy, PartialEq)]
struct ColorConfig {
    scheme: ColorScheme,
    brightness: f64,
    contrast: f64,
    saturation: f64,
    custom_r: f64,
    custom_g: f64,
    custom_b: f64,
}

/// State of the double pendulum system
#[derive(Clone, Copy)]
struct State {
    theta1: f64, // Angle of first pendulum from vertical
    omega1: f64, // Angular velocity of first pendulum
    theta2: f64, // Angle of second pendulum from vertical
    omega2: f64, // Angular velocity of second pendulum
}

impl State {
    fn new(theta1: f64, theta2: f64) -> Self {
        Self {
            theta1,
            theta2,
            omega1: 0.0, // Start at rest
            omega2: 0.0, // Start at rest
        }
    }
}

/// Computes time derivatives using Lagrangian mechanics
fn compute_derivatives(state: State, params: PendulumParams) -> State {
    let PendulumParams { m1, m2, l1, l2 } = params;

    // Precompute trigonometric values
    let delta = state.theta1 - state.theta2;
    let sin1 = state.theta1.sin();
    let sin_diff = delta.sin();
    let cos_diff = delta.cos();
    let sin_2theta2 = (state.theta1 - 2.0 * state.theta2).sin();

    // Common denominator from the mass matrix
    let denom_factor = 2.0 * m1 + m2 - m2 * (2.0 * delta).cos();
    let denom1 = l1 * denom_factor;
    let denom2 = l2 * denom_factor;

    let omega1_sq = state.omega1 * state.omega1;
    let omega2_sq = state.omega2 * state.omega2;

    // Angular acceleration of first pendulum
    let domega1 = (-G * (2.0 * m1 + m2) * sin1
        - m2 * G * sin_2theta2
        - 2.0 * sin_diff * m2 * (omega2_sq * l2 + omega1_sq * l1 * cos_diff))
        / denom1;

    // Angular acceleration of second pendulum
    let domega2 = (2.0
        * sin_diff
        * (omega1_sq * l1 * (m1 + m2)
            + G * (m1 + m2) * state.theta1.cos()
            + omega2_sq * l2 * m2 * cos_diff))
        / denom2;

    State {
        theta1: state.omega1,
        omega1: domega1,
        theta2: state.omega2,
        omega2: domega2,
    }
}

/// Fourth-order Runge-Kutta integration step
fn rk4_step(state: State, dt: f64, params: PendulumParams) -> State {
    let dt_half = dt / 2.0;
    let dt_sixth = dt / 6.0;

    // RK4 intermediate calculations
    let k1 = compute_derivatives(state, params);

    let state2 = State {
        theta1: state.theta1 + dt_half * k1.theta1,
        omega1: state.omega1 + dt_half * k1.omega1,
        theta2: state.theta2 + dt_half * k1.theta2,
        omega2: state.omega2 + dt_half * k1.omega2,
    };
    let k2 = compute_derivatives(state2, params);

    let state3 = State {
        theta1: state.theta1 + dt_half * k2.theta1,
        omega1: state.omega1 + dt_half * k2.omega1,
        theta2: state.theta2 + dt_half * k2.theta2,
        omega2: state.omega2 + dt_half * k2.omega2,
    };
    let k3 = compute_derivatives(state3, params);

    let state4 = State {
        theta1: state.theta1 + dt * k3.theta1,
        omega1: state.omega1 + dt * k3.omega1,
        theta2: state.theta2 + dt * k3.theta2,
        omega2: state.omega2 + dt * k3.omega2,
    };
    let k4 = compute_derivatives(state4, params);

    // Final weighted average
    State {
        theta1: state.theta1
            + dt_sixth * (k1.theta1 + 2.0 * k2.theta1 + 2.0 * k3.theta1 + k4.theta1),
        omega1: state.omega1
            + dt_sixth * (k1.omega1 + 2.0 * k2.omega1 + 2.0 * k3.omega1 + k4.omega1),
        theta2: state.theta2
            + dt_sixth * (k1.theta2 + 2.0 * k2.theta2 + 2.0 * k3.theta2 + k4.theta2),
        omega2: state.omega2
            + dt_sixth * (k1.omega2 + 2.0 * k2.omega2 + 2.0 * k3.omega2 + k4.omega2),
    }
}

/// Simulates the pendulum evolution from initial conditions
fn simulate_pendulum(
    initial_state: State,
    config: SimulationConfig,
    params: PendulumParams,
) -> State {
    let mut state = initial_state;
    for _ in 0..config.steps {
        state = rk4_step(state, config.dt, params);
    }
    state
}

/// Creates color based on the final pendulum state
fn state_to_color(final_state: State, color_config: ColorConfig) -> Rgb<u8> {
    let sin_theta2 = final_state.theta2.sin();
    let cos_theta2 = final_state.theta2.cos();
    let _sin_theta1 = final_state.theta1.sin();
    
    // Base intensity from pendulum state
    let intensity = ((sin_theta2 + 1.0) * 0.5).clamp(0.0, 1.0);
    let phase = ((cos_theta2 + 1.0) * 0.5).clamp(0.0, 1.0);
    let combined = (intensity + phase * 0.3).clamp(0.0, 1.0);
    
    // Apply contrast and brightness
    let adjusted = (combined * color_config.contrast + color_config.brightness).clamp(0.0, 1.0);
    
    let (r, g, b) = match color_config.scheme {
        ColorScheme::RedPurple => {
            let red = (adjusted * 0.8 + 0.2) * 255.0;
            let green = (1.0 - adjusted) * 0.3 * 255.0;
            let blue = (0.3 + adjusted * 0.4) * 255.0;
            (red, green, blue)
        },
        ColorScheme::BlueGreen => {
            let blue = (adjusted * 0.8 + 0.2) * 255.0;
            let green = (0.5 + adjusted * 0.5) * 255.0;
            let red = (1.0 - adjusted) * 0.2 * 255.0;
            (red, green, blue)
        },
        ColorScheme::Rainbow => {
            let hue = adjusted * 6.0; // 0-6 for full rainbow
            let (r, g, b) = hsv_to_rgb(hue, 0.8, 0.9);
            (r * 255.0, g * 255.0, b * 255.0)
        },
        ColorScheme::Fire => {
            let red = (adjusted * 0.9 + 0.1) * 255.0;
            let green = (adjusted * adjusted * 0.6) * 255.0;
            let blue = (adjusted * adjusted * adjusted * 0.2) * 255.0;
            (red, green, blue)
        },
        ColorScheme::Ocean => {
            let blue = (adjusted * 0.7 + 0.3) * 255.0;
            let green = (0.4 + adjusted * 0.4) * 255.0;
            let red = (0.1 + adjusted * 0.2) * 255.0;
            (red, green, blue)
        },
        ColorScheme::Sunset => {
            let red = (adjusted * 0.8 + 0.2) * 255.0;
            let green = (adjusted * 0.4 + 0.1) * 255.0;
            let blue = (adjusted * adjusted * 0.3) * 255.0;
            (red, green, blue)
        },
        ColorScheme::Custom => {
            let red = (adjusted * color_config.custom_r) * 255.0;
            let green = (adjusted * color_config.custom_g) * 255.0;
            let blue = (adjusted * color_config.custom_b) * 255.0;
            (red, green, blue)
        },
    };
    
    // Apply saturation
    let gray = (r + g + b) / 3.0;
    let final_r = (r + (gray - r) * (1.0 - color_config.saturation)).clamp(0.0, 255.0);
    let final_g = (g + (gray - g) * (1.0 - color_config.saturation)).clamp(0.0, 255.0);
    let final_b = (b + (gray - b) * (1.0 - color_config.saturation)).clamp(0.0, 255.0);

    Rgb([final_r as u8, final_g as u8, final_b as u8])
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = v - c;
    
    let (r, g, b) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    (r + m, g + m, b + m)
}

/// Renders a single frame of the fractal
fn render_frame(
    width: u32, 
    height: u32, 
    phase_offset: f64,
    pendulum: PendulumParams,
    sim_config: SimulationConfig,
    color_config: ColorConfig,
) -> RgbImage {
    let mut img = RgbImage::new(width, height);
    let two_pi = 2.0 * PI;

    // Parallel processing for each row
    let row_results: Vec<Vec<(u32, u32, Rgb<u8>)>> = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut row_pixels = Vec::with_capacity(width as usize);

            for x in 0..width {
                // Map pixel coordinates to initial angles
                let theta1 = (x as f64 / width as f64) * two_pi;
                let theta2 = (y as f64 / height as f64) * two_pi + phase_offset;

                // Create initial state and simulate
                let initial_state = State::new(theta1, theta2);
                let final_state = simulate_pendulum(initial_state, sim_config, pendulum);
                let color = state_to_color(final_state, color_config);

                row_pixels.push((x, y, color));
            }
            row_pixels
        })
        .collect();

    // Apply computed pixels to image
    for row_pixels in row_results {
        for (x, y, pixel) in row_pixels {
            img.put_pixel(x, y, pixel);
        }
    }

    img
}

/// Main application state
struct DoublePendulumApp {
    // Parameters
    pendulum: PendulumParams,
    sim_config: SimulationConfig,
    render_config: RenderConfig,
    color_config: ColorConfig,
    phase_offset: f64,
    
    // UI state
    texture: Option<TextureHandle>,
    is_rendering: bool,
    last_render_time: Option<Instant>,
    last_render_duration: Option<f64>,
    render_requested: bool,
    
    // Real-time controls
    auto_render: bool,
    animation_enabled: bool,
    animation_speed: f64,
    last_params: Option<(PendulumParams, SimulationConfig, RenderConfig, ColorConfig, f64)>,
    
    // Threading
    render_sender: Sender<RenderRequest>,
    render_receiver: Receiver<RenderResult>,
}

#[derive(Clone)]
struct RenderRequest {
    width: u32,
    height: u32,
    phase_offset: f64,
    pendulum: PendulumParams,
    sim_config: SimulationConfig,
    color_config: ColorConfig,
}

#[derive(Clone)]
struct RenderResult {
    image: RgbImage,
    render_time: f64,
}

impl Default for DoublePendulumApp {
    fn default() -> Self {
        let (render_sender, _) = bounded::<RenderRequest>(1);
        let (_, render_receiver) = bounded::<RenderResult>(1);
        
        Self {
            pendulum: PendulumParams {
                m1: 1.0,
                m2: 1.0,
                l1: 1.0,
                l2: 1.0,
            },
            sim_config: SimulationConfig {
                dt: 0.01,
                steps: 800,
            },
            render_config: RenderConfig {
                width: 400,
                height: 400,
                preview_quality: true,
            },
            color_config: ColorConfig {
                scheme: ColorScheme::RedPurple,
                brightness: 0.0,
                contrast: 1.0,
                saturation: 1.0,
                custom_r: 1.0,
                custom_g: 0.3,
                custom_b: 0.7,
            },
            phase_offset: 0.0,
            texture: None,
            is_rendering: false,
            last_render_time: None,
            last_render_duration: None,
            render_requested: false,
            auto_render: true,
            animation_enabled: false,
            animation_speed: 0.1,
            last_params: None,
            render_sender,
            render_receiver,
        }
    }
}

impl eframe::App for DoublePendulumApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update animation
        if self.animation_enabled {
            self.phase_offset += self.animation_speed * 0.016; // ~60fps
            if self.phase_offset > 2.0 * PI {
                self.phase_offset -= 2.0 * PI;
            }
        }

        // Check for completed renders
        if let Ok(result) = self.render_receiver.try_recv() {
            self.is_rendering = false;
            self.last_render_time = Some(Instant::now());
            self.last_render_duration = Some(result.render_time);
            
            // Convert to egui texture
            let size = [result.image.width() as usize, result.image.height() as usize];
            let pixels: Vec<egui::Color32> = result.image
                .pixels()
                .map(|p| egui::Color32::from_rgb(p[0], p[1], p[2]))
                .collect();
            
            let color_image = ColorImage::from_rgba_unmultiplied(size, 
                &pixels.iter().flat_map(|c| [c.r(), c.g(), c.b(), c.a()]).collect::<Vec<u8>>());
            self.texture = Some(ctx.load_texture("pendulum_fractal", color_image, Default::default()));
        }

        // Check if parameters changed for auto-render
        let current_params = (self.pendulum, self.sim_config, self.render_config, self.color_config, self.phase_offset);
        let params_changed = self.last_params.map_or(true, |last| last != current_params);
        
        if params_changed {
            self.last_params = Some(current_params);
            if self.auto_render {
                self.render_requested = true;
            }
        }

        // Request render if needed
        if self.render_requested && !self.is_rendering {
            let steps = if self.render_config.preview_quality { 100 } else { 800 };
            let sim_config = SimulationConfig {
                dt: self.sim_config.dt,
                steps,
            };
            
            let request = RenderRequest {
                width: self.render_config.width,
                height: self.render_config.height,
                phase_offset: self.phase_offset,
                pendulum: self.pendulum,
                sim_config,
                color_config: self.color_config,
            };
            
            if self.render_sender.try_send(request).is_ok() {
                self.is_rendering = true;
                self.render_requested = false;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Double Pendulum Fractal Generator");
            
            ui.horizontal(|ui| {
                // Left panel - Controls
                ui.vertical(|ui| {
                    ui.group(|ui| {
                        ui.heading("Pendulum Parameters");
                        
                        ui.add(egui::Slider::new(&mut self.pendulum.m1, 0.1..=5.0).text("Mass 1 (kg)"));
                        ui.add(egui::Slider::new(&mut self.pendulum.m2, 0.1..=5.0).text("Mass 2 (kg)"));
                        ui.add(egui::Slider::new(&mut self.pendulum.l1, 0.1..=3.0).text("Length 1 (m)"));
                        ui.add(egui::Slider::new(&mut self.pendulum.l2, 0.1..=3.0).text("Length 2 (m)"));
                    });
                    
                    ui.add_space(10.0);
                    
                    ui.group(|ui| {
                        ui.heading("Simulation Settings");
                        
                        ui.add(egui::Slider::new(&mut self.sim_config.dt, 0.001..=0.1).text("Time Step"));
                        ui.add(egui::Slider::new(&mut self.phase_offset, 0.0..=2.0 * PI).text("Phase Offset"));
                        
                        ui.checkbox(&mut self.render_config.preview_quality, "Preview Quality (faster)");
                    });
                    
                    ui.add_space(10.0);
                    
                    ui.group(|ui| {
                        ui.heading("Color Settings");
                        
                        // Color scheme selection
                        ui.horizontal(|ui| {
                            ui.label("Scheme:");
                            egui::ComboBox::from_id_source("color_scheme")
                                .selected_text(format!("{:?}", self.color_config.scheme))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut self.color_config.scheme, ColorScheme::RedPurple, "Red Purple");
                                    ui.selectable_value(&mut self.color_config.scheme, ColorScheme::BlueGreen, "Blue Green");
                                    ui.selectable_value(&mut self.color_config.scheme, ColorScheme::Rainbow, "Rainbow");
                                    ui.selectable_value(&mut self.color_config.scheme, ColorScheme::Fire, "Fire");
                                    ui.selectable_value(&mut self.color_config.scheme, ColorScheme::Ocean, "Ocean");
                                    ui.selectable_value(&mut self.color_config.scheme, ColorScheme::Sunset, "Sunset");
                                    ui.selectable_value(&mut self.color_config.scheme, ColorScheme::Custom, "Custom");
                                });
                        });
                        
                        // Color adjustments
                        ui.add(egui::Slider::new(&mut self.color_config.brightness, -1.0..=1.0).text("Brightness"));
                        ui.add(egui::Slider::new(&mut self.color_config.contrast, 0.1..=3.0).text("Contrast"));
                        ui.add(egui::Slider::new(&mut self.color_config.saturation, 0.0..=2.0).text("Saturation"));
                        
                        // Custom color controls
                        if self.color_config.scheme == ColorScheme::Custom {
                            ui.add_space(5.0);
                            ui.label("Custom Colors:");
                            ui.add(egui::Slider::new(&mut self.color_config.custom_r, 0.0..=1.0).text("Red"));
                            ui.add(egui::Slider::new(&mut self.color_config.custom_g, 0.0..=1.0).text("Green"));
                            ui.add(egui::Slider::new(&mut self.color_config.custom_b, 0.0..=1.0).text("Blue"));
                        }
                    });
                    
                    ui.add_space(10.0);
                    
                    ui.group(|ui| {
                        ui.heading("Render Settings");
                        
                        ui.add(egui::Slider::new(&mut self.render_config.width, 100..=800).text("Width"));
                        ui.add(egui::Slider::new(&mut self.render_config.height, 100..=800).text("Height"));
                        
                        ui.add_space(5.0);
                        ui.checkbox(&mut self.auto_render, "Auto Render (Real-time)");
                        ui.checkbox(&mut self.animation_enabled, "Animation");
                        
                        if self.animation_enabled {
                            ui.add(egui::Slider::new(&mut self.animation_speed, 0.01..=1.0).text("Animation Speed"));
                        }
                        
                        ui.add_space(5.0);
                        if ui.button("Render Now").clicked() {
                            self.render_requested = true;
                        }
                    });
                    
                    ui.add_space(10.0);
                    
                    if let Some(last_time) = self.last_render_time {
                        ui.label(format!("Last render: {:.2}s ago", last_time.elapsed().as_secs_f64()));
                    }

                    if let Some(duration) = self.last_render_duration {
                        ui.label(format!("Render time: {:.2}s", duration));
                    }
                    
                    if self.is_rendering {
                        ui.label("Rendering...");
                        ui.spinner();
                    } else if self.auto_render {
                        ui.label("🔄 Real-time mode active");
                    }
                    
                    if self.animation_enabled {
                        ui.label(format!("🎬 Animating (speed: {:.2})", self.animation_speed));
                    }
                });
                
                ui.add_space(20.0);
                
                // Right panel - Image display
                ui.vertical(|ui| {
                    ui.heading("Fractal Preview");
                    
                    if let Some(texture) = &self.texture {
                        let max_size = ui.available_size().min(egui::Vec2::splat(400.0));
                        ui.add(egui::Image::new(texture).max_size(max_size));
                    } else {
                        ui.label("No image rendered yet. Adjust parameters and click 'Render Now'.");
                    }
                });
            });
        });
        
        // Request continuous updates for real-time rendering
        if self.auto_render || self.animation_enabled {
            ctx.request_repaint();
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    // Start render thread
    let (render_sender, render_receiver) = bounded::<RenderRequest>(1);
    let (result_sender, result_receiver) = bounded::<RenderResult>(1);
    
    thread::spawn(move || {
        while let Ok(request) = render_receiver.recv() {
            let start_time = Instant::now();
            
            let image = render_frame(
                request.width,
                request.height,
                request.phase_offset,
                request.pendulum,
                request.sim_config,
                request.color_config,
            );
            
            let render_time = start_time.elapsed().as_secs_f64();
            
            let result = RenderResult {
                image,
                render_time,
            };
            
            let _ = result_sender.send(result);
        }
    });

    let mut app = DoublePendulumApp::default();
    app.render_sender = render_sender;
    app.render_receiver = result_receiver;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Double Pendulum Fractal Generator",
        options,
        Box::new(|_cc| Ok(Box::new(app))),
    )
}