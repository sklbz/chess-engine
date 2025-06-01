use chess::board::board::Board;
use chess::board::nnue_input_vector::VectorOutput;
use chess::legal_moves;
use chess::legal_moves::generate_possible_moves::generate_move_vec;
use chess::legal_moves::misc::Color;
use multilayer_perceptron::mlp::multilayer_perceptron::MultilayerPerceptron;

use crate::move_to_number::move_hash;

struct ChessEngine {
    mlp: MultilayerPerceptron,
}

impl ChessEngine {
    fn new() -> ChessEngine {
        // Inspiration from Stockfish NNUE architecture
        let architecture = vec![768, 1024, 1536, 1792];

        let engine = MultilayerPerceptron::new(architecture);

        ChessEngine { mlp: engine }
    }

    fn predict(&self, board: Board, color: &Color) -> Move {
        let input = board.to_vector();

        let raw_output = self.mlp.calc(input);

        let legal_moves = board.get_legal_moves(color);
        let moves_indices = legal_moves
            .iter()
            .map(|mv| move_hash(mv))
            .collect::<Vec<u32>>();

        let trimmed_output = raw_output
            .iter()
            .enumerate()
            .filter(|(i, _)| moves_indices.contains(i));
    }
}

fn main() {
    let mut board = Board::init();
}
