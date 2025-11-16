mod window;

fn main() {
    println!("Hello, world!");
    pollster::block_on(window::run());
}
