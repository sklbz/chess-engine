use chess::board::board::Board;
use chess::board::nnue_input_vector::VectorOutput;
use chess::legal_moves::misc::Color;
use chess::utils::move_to_string;
use multilayer_perceptron::mlp::multilayer_perceptron::*;

use super::softmax::Softmax;

use crate::move_to_number::move_hash;

struct ChessEngine {
    mlp: MultiLayerPerceptron,
}

impl ChessEngine {
    fn new() -> ChessEngine {
        // Inspiration from Stockfish NNUE architecture
        let architecture = vec![768, 1024, 1536, 1792];

        let engine = MultiLayerPerceptron::new(architecture);

        ChessEngine { mlp: engine }
    }

    fn predict(&self, board: Board, color: &Color) -> &str {
        let input: Vec<f64> = board.to_vector();

        let raw_output: Vec<f64> = self.mlp.calc(input);

        let legal_moves = board.get_legal_moves(color);
        let moves_indices = legal_moves
            .iter()
            .map(|mv| move_hash(&move_to_string(mv)))
            .collect::<Vec<usize>>();

        let trimmed_output = raw_output
            .iter()
            .enumerate()
            .filter(|(i, _)| moves_indices.contains(i))
            .map(|(_, x)| *x)
            .collect::<Vec<f64>>()
            .softmax(1.0);

        "a1a1"
    }
}
