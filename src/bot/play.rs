use crate::{nn::engine::ChessEngine, utils::move_to_output::move_vec};
use chess::{
    board::{board::Board, nnue_input_vector::VectorOutput},
    legal_moves::{is_move_possible::is_possible, misc::Color},
    utils::{string_is_move, string_to_move, user_input},
};
use multilayer_perceptron::mlp::utils::{Database, SharedVector};

use std::env::args;

type SharedBiVector = (SharedVector, SharedVector);

fn add_game_to_data(
    data: &mut Database,
    white: &[SharedBiVector],
    black: &[SharedBiVector],
    coef: f64,
) {
    for (i, (input, output)) in white.iter().rev().enumerate() {
        let weight = (-(i as f64)).exp();
        let sample = (input.clone(), output.clone(), coef * weight);
        data.push(sample);
    }
    for (i, (input, output)) in black.iter().rev().enumerate() {
        let weight = (-(i as f64)).exp();
        let sample = (input.clone(), output.clone(), -coef * weight);
        data.push(sample);
    }
}

pub fn run() {
    let args: Vec<String> = args().collect();

    let mut board = Board::init();
    let mut data: Database = Vec::new();
    let mut white_moves: Vec<SharedBiVector> = Vec::new();
    let mut black_moves: Vec<SharedBiVector> = Vec::new();
    let mut engine = ChessEngine::new();

    engine.load_model("elite.model.json");

    if args.len() > 1 {
        println!("Training");
        engine.train_from_file(&args[1]);
    }

    loop {
        board.display();

        let input = user_input();

        if input.is_empty() {
            println!("No commands received.");
            continue;
        }

        if input == "reset" {
            board = Board::init();
            white_moves = Vec::new();
            black_moves = Vec::new();
            continue;
        }

        if input == "quit" {
            break;
        }

        let mut input_split = input.split_whitespace();
        let command = match input_split.next() {
            Some(value) => value,
            None => {
                println!("Empty command");
                continue;
            }
        };
        let option = input_split.last().unwrap_or_default();

        match command {
            "train" => {
                let path = option;
                engine.train_from_file(path);
                continue;
            }
            "save" => {
                engine.save_model();
                continue;
            }
            "load" => {
                let path = option;
                engine.load_model(path);
                println!("Parameters loaded from {path}");
                continue;
            }
            "temp" => {
                let temp = match option {
                    "up" => engine.temp_up(),
                    "down" => engine.temp_down(),
                    _ => engine.get_temp(),
                };
                println!("Temperature: {}", temp);
                continue;
            }
            _ => (),
        };

        if !string_is_move(command) {
            println!("Input does not match the UCI format");
            continue;
        }

        if !is_possible(&board, &string_to_move(command), Color::White) {
            println!("Invalid move");
            continue;
        }

        board.make_move_str(command);
        white_moves.push((board.to_vector().into(), move_vec(input).into()));
        println!();

        if board.is_checkmate(Color::Black) {
            board.display();
            println!("White win by checkmate");
            board = Board::init();

            add_game_to_data(&mut data, &white_moves, &black_moves, 1.);

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
        black_moves.push((board.to_vector().into(), move_vec(engine_move).into()));

        if board.is_checkmate(Color::White) {
            board.display();
            println!("Black win by checkmate");
            board = Board::init();

            add_game_to_data(&mut data, &white_moves, &black_moves, -1.);

            white_moves = Vec::new();
            black_moves = Vec::new();

            engine.train_from_samples(&data);
            continue;
        }
    }
}
