mod bot;
mod database;
mod nn;
mod rl;
mod train;
mod utils;

// use bot::play::run;
use nn::engine::ChessEngine;
// use rl::data_generation::run;

fn main() {
    let mut engine = ChessEngine::new();

    engine.load_model("elite.model.json");
    engine.auto_train("../data/lichess-elite-short");
}
