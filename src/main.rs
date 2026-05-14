mod bot;
mod nn;
mod train;
mod utils;

use bot::play::run;
use nn::engine::ChessEngine;
use train::carlsen::train;

use std::env::args;

fn main() {
    run()
}

fn main_training() {
    let mut model = ChessEngine::new();
    model.train_from_file("../data/lichess-elite-short");
    // model.save_model("models/lichess-elite-complex");

    let args: Vec<String> = args().collect();
    if args.len() > 1 {
        train();
    } else {
        run();
    }
}
