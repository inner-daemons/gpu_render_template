struct FragmentUniforms {
    size: vec2<f32>,
    mouse: vec2<f32>
};


@group(0) @binding(0) var<uniform> frag_uniforms : FragmentUniforms;

@fragment
fn main(@builtin(position) clip_pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}