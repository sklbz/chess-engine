use super::stockfish_handler::{Evaluation, StockfishProcess};
use crate::nn::engine::ChessEngine;
use Evaluation::*;
use chess::{
    board::{board::Board, fen_handling::FenHandling},
    legal_moves::{is_move_possible::is_possible, misc::Color::*},
    utils::string_to_move,
};
use multilayer_perceptron::mlp::utils::{Database, SharedVector};

pub struct EvaluationHolder {
    stockfish: StockfishProcess,
    data: Vec<Evaluation>,
}

impl EvaluationHolder {
    fn with_capacity(capacity: usize) -> EvaluationHolder {
        let stockfish = match StockfishProcess::spawn() {
            Ok(process) => process,
            Err(e) => panic!("{e}"),
        };
        let data: Vec<Evaluation> = Vec::with_capacity(capacity);
        EvaluationHolder { stockfish, data }
    }

    fn eval(&mut self, fen: &str) -> Evaluation {
        match self.stockfish.evaluate_fen(fen, 8) {
            Ok(eval) => {
                self.data.push(eval);
                eval
            }
            Err(e) => panic!("{e}"),
        }
    }

    fn choose_reward(&self, states: &[SharedVector], moves: &[SharedVector]) -> Database {
        assert_eq!(self.data.len(), states.len());
        assert_eq!(states.len(), moves.len());
        let data: Database = Vec::new();
        return data;
    }
}

pub fn run() {
    let mut game: String = String::new();
    let mut board: Board = Board::init();
    // let fen = "4k3/7R/8/8/8/3pp3/3npn2/R2nKn2 w Q - 0 1";
    // let mut board: Board = Board::from_fen(fen);
    let mut evals = EvaluationHolder::with_capacity(101);
    let mut engine: ChessEngine = ChessEngine::new();
    let mut turn = White;

    engine.load_model("elite.model.json");

    print!("\x1B[2J");

    for i in 0..100 {
        print!("\x1B[1;1H");
        match i {
            0 | 1 => println!("{i} ply"),
            _ => println!("{i} plies"),
        };

        let eval = evals.eval(&board.to_fen(turn));
        match eval {
            Centipawns(cp) => println!("Evaluation: {:.2}", cp as f32 / 100.0),
            Mate(n) => println!("Mate in {}", n),
        };

        board.display();

        let engine_move = engine.predict_symm(&board, &turn);

        if !is_possible(&board, &string_to_move(&engine_move), turn) {
            panic!("Engine made invalid move: {}", engine_move);
        }

        board.make_move_str(engine_move.as_str());
        game = format!("{game} {engine_move}");

        if board.is_checkmate(turn) {
            let color: &str = match turn {
                White => "White",
                Black => "Black",
                Null => "Null",
            };
            println!("{color} win by checkmate");
            break;
        }

        if board.is_stalemate() {
            break;
            // println!("\r Starting a new game");
            // board = Board::from_fen(fen);
            // game = String::new();
            // turn = White;
            // continue;
        }

        turn = !turn;
    }
    evals.eval(&board.to_fen(turn));

    board.display();
    println!("{game}");
    println!("{:?}", evals.data);
}
