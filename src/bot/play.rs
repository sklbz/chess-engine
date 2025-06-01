use crate::nn::engine::ChessEngine;
use chess::{board::board::Board, legal_moves::misc::Color, utils::user_input};

pub fn run() {
    let mut board = Board::init();
    let engine = ChessEngine::new();

    loop {
        board.display();

        let _ = board.make_move_str(user_input().as_str());

        if board.is_checkmate(Color::Black) {
            board.display();
            println!("White win by checkmate");
            break;
        }

        let _ = board.make_move_str(engine.predict(&board, &Color::Black).as_str());

        if board.is_checkmate(Color::White) {
            board.display();
            println!("Black win by checkmate");
            break;
        }
    }
}
