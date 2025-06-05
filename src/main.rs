mod bot;
mod nn;
mod train;
mod utils;

use bot::play::run;
use train::carlsen::train;

use std::env::args;

fn main() {
    let args: Vec<String> = args().collect();
    if args.len() > 1 {
        train();
    } else {
        run();
    }
}
