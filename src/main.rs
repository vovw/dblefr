use gif::{Encoder as GifEncoder, Frame as GifFrame, Repeat};
use image::{Rgb, RgbImage};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;

const G: f64 = 9.81; // Gravitational acceleration (m/s²)

#[derive(Clone, Copy)]
struct PendulumParams {
    m1: f64, // Mass of first pendulum (kg)
    m2: f64, // Mass of second pendulum (kg)
    l1: f64, // Length of first pendulum (m)
    l2: f64, // Length of second pendulum (m)
}

#[derive(Clone, Copy)]
struct SimulationConfig {
    dt: f64,      // Time step for numerical integration (s)
    steps: usize, // Number of RK4 integration steps per pixel
}

// Default simulation parameters
const PENDULUM: PendulumParams = PendulumParams {
    m1: 1.0,
    m2: 1.0,
    l1: 1.0,
    l2: 1.0,
};

const SIMULATION: SimulationConfig = SimulationConfig {
    dt: 0.01,
    steps: 800,
};

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
fn state_to_color(final_state: State) -> Rgb<u8> {
    let sin_theta2 = final_state.theta2.sin();
    let intensity = ((sin_theta2 + 1.0) * 0.5).clamp(0.0, 1.0);

    // Elegant red-purple gradient
    let red = ((intensity * 0.8 + 0.2) * 255.0) as u8;
    let green = ((1.0 - intensity) * 0.3 * 255.0) as u8;
    let blue = ((0.3 + intensity * 0.4) * 255.0) as u8;

    Rgb([red.max(50), green.max(20), blue.max(10)])
}

/// Renders a single frame of the fractal
fn render_frame(width: u32, height: u32, phase_offset: f64) -> RgbImage {
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
                let final_state = simulate_pendulum(initial_state, SIMULATION, PENDULUM);
                let color = state_to_color(final_state);

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

/// Animation configuration
struct AnimationConfig {
    width: u32,
    height: u32,
    frames: u32,
    fps: u16,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            width: 320,
            height: 320,
            frames: 60,
            fps: 20,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnimationConfig::default();
    let phase_step = 2.0 * PI / config.frames as f64;
    let frame_delay = 100u16 / config.fps;

    // Set up GIF encoder
    let mut file = File::create("double_pendulum_anim.gif")?;
    let mut encoder = GifEncoder::new(&mut file, config.width as u16, config.height as u16, &[])?;
    encoder.set_repeat(Repeat::Infinite)?;

    println!(
        "Generating {} frames at {}x{} resolution...",
        config.frames, config.width, config.height
    );

    // Generate animation frames
    for frame_num in 0..config.frames {
        let phase = frame_num as f64 * phase_step;
        eprintln!("Rendering frame {}/{}", frame_num + 1, config.frames);

        let frame_img = render_frame(config.width, config.height, phase);

        // Convert to RGBA for GIF
        let mut rgba_buffer = Vec::with_capacity((config.width * config.height * 4) as usize);
        for pixel in frame_img.pixels() {
            rgba_buffer.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
        }

        let mut gif_frame = GifFrame::from_rgba_speed(
            config.width as u16,
            config.height as u16,
            &mut rgba_buffer,
            10,
        );
        gif_frame.delay = frame_delay;
        encoder.write_frame(&gif_frame)?;
    }

    println!("\n✨ Successfully created double_pendulum_anim.gif");
    println!("This fractal reveals the chaotic beauty of double pendulum dynamics!");

    Ok(())
}
