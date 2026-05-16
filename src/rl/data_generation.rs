use crate::nn::engine::ChessEngine;
use chess::{
    board::{board::Board, fen_handling::FenHandling},
    legal_moves::{is_move_possible::is_possible, misc::Color::*},
    utils::string_to_move,
};

pub fn run() {
    let mut game: String = String::new();
    // let mut board: Board = Board::init();
    let fen = "4k3/7R/8/8/8/3pp3/3npn2/R2nKn2 w Q - 0 1";
    let mut board: Board = Board::from_fen(fen);
    let mut engine: ChessEngine = ChessEngine::new();
    let mut turn = White;

    engine.load_model("carlsen.elite.model.json");

    loop {
        board.display();
        // print!("\r Game length: {}", game.len());
        // stdout().flush().unwrap();

        let engine_move = engine.predict(&board, &turn);

        if !is_possible(&board, &string_to_move(&engine_move), turn) {
            panic!("Engine made invalid move: {}", engine_move);
        }

        board.make_move_str(engine_move.as_str());
        game = format!("{game} {engine_move}");

        if board.is_checkmate(turn) {
            board.display();
            let color: &str = match turn {
                White => "White",
                Black => "Black",
                Null => "Null",
            };
            println!("{color} win by checkmate");
            println!("{game}");
            return;
        }

        if board.is_stalemate() {
            println!("\r Starting a new game");
            board = Board::from_fen(fen);
            game = String::new();
            turn = White;
            continue;
        }

        turn = !turn;
    }
}
