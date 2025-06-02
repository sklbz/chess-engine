use crate::nn::engine::ChessEngine;
use chess::{
    board::board::Board,
    legal_moves::{is_move_possible::is_possible, misc::Color},
    utils::{string_to_move, user_input},
};

pub fn run() {
    let mut board = Board::init();
    let engine = ChessEngine::new();

    loop {
        board.display();

        let input = user_input();

        if input == "quit" {
            break;
        }

        while !is_possible(&board, &string_to_move(&input), Color::White) {
            println!("Invalid move");
            continue;
        }

        let _ = board.make_move_str(user_input().as_str());

        if board.is_checkmate(Color::Black) {
            board.display();
            println!("White win by checkmate");
        }

        let _ = board.make_move_str(engine.predict(&board, &Color::Black).as_str());

        if board.is_checkmate(Color::White) {
            board.display();
            println!("Black win by checkmate");
        }
    }
}
