use crate::nn::engine::ChessEngine;
use chess::{
    board::board::Board,
    legal_moves::{is_move_possible::is_possible, misc::Color},
    utils::{string_to_move, user_input},
};

use std::env::args;

pub fn run() {
    let args: Vec<String> = args().collect();

    let mut board = Board::init();
    let mut engine = ChessEngine::new();

    if args.len() > 1 {
        println!("Training");
        engine.train_from_file(&args[1]);
    }

    loop {
        board.display();

        let input = user_input();

        if input == "quit" {
            break;
        }

        if !is_possible(&board, &string_to_move(&input), Color::White) {
            println!("Invalid move");
            continue;
        }

        board.make_move_str(input.as_str());
        println!();

        if board.is_checkmate(Color::Black) {
            board.display();
            println!("White win by checkmate");
            break;
        }

        let engine_move = engine.predict(&board, &Color::Black);

        if !is_possible(&board, &string_to_move(&engine_move), Color::Black) {
            panic!("Engine made invalid move");
        }

        board.make_move_str(engine_move.as_str());

        if board.is_checkmate(Color::White) {
            board.display();
            println!("Black win by checkmate");
            break;
        }
    }
}
