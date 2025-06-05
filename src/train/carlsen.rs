use crate::nn::engine::ChessEngine;

use std::env::args;

pub fn run() {
    let mut engine = ChessEngine::new();

    let args: Vec<String> = args().collect();
    if args.len() > 3 {
        engine.load_model(&args[3]);
    }

    if args.len() > 2 {
        engine.train_from_file(&args[2]);
        engine.save_model(&args[1]);
    }
}
