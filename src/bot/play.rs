use crate::{nn::engine::ChessEngine, utils::move_to_output::move_vec};
use chess::{
    board::{board::Board, nnue_input_vector::VectorOutput},
    legal_moves::{is_move_possible::is_possible, misc::Color},
    utils::{string_to_move, user_input},
};
use multilayer_perceptron::mlp::utils::Database;

use std::env::args;

pub fn run() {
    let args: Vec<String> = args().collect();

    let mut board = Board::init();
    let mut data: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let mut white_moves: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut black_moves: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut engine = ChessEngine::new();

    // engine.load_model("models/RL.model.json");
    engine.train_from_file("../data/carlsen.txt");
    if args.len() > 1 {
        println!("Training");
        engine.train_from_file(&args[1]);
    }

    loop {
        board.display();

        let input = user_input();

        if input == "reset" {
            board = Board::init();
            white_moves = Vec::new();
            black_moves = Vec::new();
            continue;
        }

        if input == "quit" {
            break;
        }

        if input.split_whitespace().next().unwrap() == "train" {
            let path = input.split_whitespace().last().unwrap();
            engine.train_from_file(path);
            continue;
        }

        if input.split_whitespace().next().unwrap() == "save" {
            engine.save_model();
            continue;
        }

        if input.split_whitespace().next().unwrap() == "load" {
            let path = input.split_whitespace().last().unwrap();
            engine.load_model(path);
            println!("Parameters loaded from {path}");
            continue;
        }

        if !is_possible(&board, &string_to_move(&input), Color::White) {
            println!("Invalid move");
            continue;
        }

        board.make_move_str(input.as_str());
        white_moves.push((board.to_vector(), move_vec(input)));
        println!();

        if board.is_checkmate(Color::Black) {
            board.display();
            println!("White win by checkmate");
            board = Board::init();

            for value in white_moves.iter() {
                let (input, output) = value;
                let sample = (input.clone(), output.clone(), 1.0);
                data.push(sample);
            }
            for value in black_moves.iter() {
                let (input, output) = value;
                let sample = (input.clone(), output.clone(), -1.0);
                data.push(sample);
            }
            white_moves = Vec::new();
            black_moves = Vec::new();

            engine.train_from_samples(&data);
            continue;
        }

        let engine_move = engine.predict(&board, &Color::Black);

        if !is_possible(&board, &string_to_move(&engine_move), Color::Black) {
            panic!("Engine made invalid move");
        }

        board.make_move_str(engine_move.as_str());
        white_moves.push((board.to_vector(), move_vec(engine_move)));

        if board.is_checkmate(Color::White) {
            board.display();
            println!("Black win by checkmate");
            board = Board::init();
            for value in white_moves.iter() {
                let (input, output) = value;
                let sample = (input.clone(), output.clone(), -1.0);
                data.push(sample);
            }
            for value in black_moves.iter() {
                let (input, output) = value;
                let sample = (input.clone(), output.clone(), 1.0);
                data.push(sample);
            }
            white_moves = Vec::new();
            black_moves = Vec::new();

            engine.train_from_samples(&data);
            continue;
        }
    }
}
